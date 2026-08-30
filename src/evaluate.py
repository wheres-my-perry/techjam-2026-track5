"""Robustness evaluation harness.

Runs a model over the full contest transform grid and reports, per condition:
AUROC, balanced accuracy at a threshold frozen on clean data, FPR@95%TPR.
Also dumps top-K most confident errors (false positives / false negatives).

Usage:
    python -m src.evaluate --manifest data/manifests/val.csv --model random \
        --out outputs/eval_random [--limit 500] [--topk 20]

Outputs:
    <out>/results.json          all metrics, machine-readable
    <out>/robustness_table.md   deliverable #4 table
    <out>/errors_clean.json     top-K FP / FN on clean data (deliverable #5 feed)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_image, load_manifest
from .metrics import condition_report, pick_threshold
from .model import load_model
from .transforms import EVAL_GRID

BATCH = 32


def _score_condition(model, samples, tf, batch=BATCH):
    scores = []
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        imgs = [tf(load_image(s.path)) for s in chunk]
        scores.extend(model.predict(imgs).tolist())
    return np.asarray(scores, dtype=np.float32)


def evaluate(model, samples, topk=20, threshold=None, conditions=None):
    import sys
    import time

    y = np.asarray([s.label for s in samples])
    results, all_scores = {}, {}
    grid = [(n, tf) for n, tf in EVAL_GRID if not conditions or n in conditions]
    if conditions:
        assert grid[0][0] == "clean", "--conditions must include clean"

    t0 = time.time()
    for i, (name, tf) in enumerate(grid, 1):
        t = time.time()
        all_scores[name] = _score_condition(model, samples, tf)
        done, total = time.time() - t, time.time() - t0
        eta = total / i * (len(grid) - i)
        print(f"[{i:2d}/{len(grid)}] {name:14s} {done:5.1f}s  (eta ~{eta:4.0f}s)",
              file=sys.stderr, flush=True)

    # threshold: the FIXED product cut-off when given (Thinh's rule: a finished product has one
    # line, and every number is read at that line); otherwise the legacy Youden pick on the
    # clean scores of this very set (threshold-free comparison only, never a product number).
    thr = float(threshold) if threshold is not None else pick_threshold(y, all_scores["clean"])
    for name, _ in grid:
        r = condition_report(y, all_scores[name], thr)
        sc = all_scores[name]
        r["tpr@thr"] = float((sc[y == 1] >= thr).mean()) if (y == 1).any() else float("nan")
        r["fpr@thr"] = float((sc[y == 0] >= thr).mean()) if (y == 0).any() else float("nan")
        r["acc@thr"] = float(((sc >= thr) == (y == 1)).mean())
        results[name] = r

    aurocs = [results[n]["auroc"] for n, _ in grid if n != "clean"]
    tprs = [results[n]["tpr@thr"] for n, _ in grid if n != "clean"]
    fprs = [results[n]["fpr@thr"] for n, _ in grid if n != "clean"]
    summary = {
        "threshold_frozen_on_clean": thr,
        "threshold_source": "fixed" if threshold is not None else "youden_on_this_set",
        "clean_tpr@thr": results["clean"]["tpr@thr"],
        "clean_fpr@thr": results["clean"]["fpr@thr"],
        "mean_transformed_tpr@thr": float(np.nanmean(tprs)),
        "mean_transformed_fpr@thr": float(np.nanmean(fprs)),
        "worst_transformed_tpr@thr": float(np.nanmin(tprs)),
        "worst_transformed_fpr@thr": float(np.nanmax(fprs)),
        "clean_auroc": results["clean"]["auroc"],
        "mean_transformed_auroc": float(np.nanmean(aurocs)),
        "worst_transformed_auroc": float(np.nanmin(aurocs)),
        "worst_condition": min(
            (n for n, _ in grid if n != "clean"), key=lambda n: results[n]["auroc"]
        ),
    }

    # per-generator breakdown: each generator's fakes vs ALL reals, per condition.
    # This is where a held-out (unseen-in-training) generator shows its number.
    generators = sorted({s.generator for s in samples if s.label == 1 and s.generator})
    per_generator = {}
    if len(generators) > 1 or (generators and generators[0]):
        real_idx = np.where(y == 0)[0]
        from .metrics import auroc as _auroc
        for g in generators:
            g_idx = np.asarray([i for i, s in enumerate(samples)
                                if s.label == 1 and s.generator == g])
            sel = np.concatenate([real_idx, g_idx])
            y_sel = y[sel]
            aurocs_g = {name: _auroc(y_sel, all_scores[name][sel]) for name, _ in grid}
            tf_vals = [v for k, v in aurocs_g.items() if k != "clean"]
            catch_g = {name: float((all_scores[name][g_idx] >= thr).mean()) for name, _ in grid}
            tf_catch = [v for k, v in catch_g.items() if k != "clean"]
            per_generator[g] = {
                "n_fake": int(len(g_idx)),
                "clean_auroc": aurocs_g["clean"],
                "mean_transformed_auroc": float(np.nanmean(tf_vals)),
                "worst_transformed_auroc": float(np.nanmin(tf_vals)),
                "conditions": aurocs_g,
                "clean_catch@thr": catch_g["clean"],
                "mean_transformed_catch@thr": float(np.nanmean(tf_catch)),
                "worst_transformed_catch@thr": float(np.nanmin(tf_catch)),
                "catch@thr": catch_g,
            }

    # error analysis on clean: most confident mistakes at the frozen threshold
    s = all_scores["clean"]
    fp = [(samples[i].path, float(s[i])) for i in np.argsort(-s) if y[i] == 0 and s[i] >= thr][:topk]
    fn = [(samples[i].path, float(s[i])) for i in np.argsort(s) if y[i] == 1 and s[i] < thr][:topk]
    errors = {"false_positives": fp, "false_negatives": fn}
    scores_dump = {"paths": np.array([s_.path for s_ in samples]), "labels": y,
                   "generators": np.array([s_.generator or "" for s_ in samples]),
                   "threshold": thr, **{f"score_{n}": all_scores[n] for n, _ in grid}}

    return results, summary, errors, per_generator, scores_dump


def to_markdown(results, summary, model_name, per_generator=None):
    lines = [
        f"# Robustness evaluation — model `{model_name}`",
        "",
        f"Threshold: {summary['threshold_frozen_on_clean']:.4f} ({summary.get('threshold_source', 'youden_on_this_set')})",
        "",
        "| condition | AUROC | fakes caught @thr | reals flagged @thr | accuracy @thr | bal.acc @thr | FPR@95%TPR | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['auroc']:.4f} | {r['tpr@thr'] * 100:.1f}% | {r['fpr@thr'] * 100:.1f}% "
            f"| {r['acc@thr'] * 100:.1f}% | {r['balanced_acc@frozen_thr']:.4f} "
            f"| {r['fpr@tpr95']:.4f} | {r['n']} |"
        )
    lines += [
        "",
        f"**Clean AUROC:** {summary['clean_auroc']:.4f} · "
        f"**Mean transformed:** {summary['mean_transformed_auroc']:.4f} · "
        f"**Worst:** {summary['worst_transformed_auroc']:.4f} ({summary['worst_condition']})",
        "",
        f"At the threshold — clean: {summary['clean_tpr@thr'] * 100:.1f}% of fakes caught, "
        f"{summary['clean_fpr@thr'] * 100:.1f}% of reals flagged · mean over transforms: "
        f"{summary['mean_transformed_tpr@thr'] * 100:.1f}% caught, {summary['mean_transformed_fpr@thr'] * 100:.1f}% flagged · "
        f"worst transform: {summary['worst_transformed_tpr@thr'] * 100:.1f}% caught, "
        f"{summary['worst_transformed_fpr@thr'] * 100:.1f}% flagged",
        "",
    ]
    if per_generator:
        lines += [
            "## Per-generator (each generator's fakes vs all reals)",
            "",
            "| generator | n fake | clean AUROC | mean transformed | worst | caught @thr clean | caught @thr mean TF | caught @thr worst TF |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for g, r in sorted(per_generator.items()):
            lines.append(
                f"| {g} | {r['n_fake']} | {r['clean_auroc']:.4f} "
                f"| {r['mean_transformed_auroc']:.4f} | {r['worst_transformed_auroc']:.4f} "
                f"| {r['clean_catch@thr'] * 100:.1f}% | {r['mean_transformed_catch@thr'] * 100:.1f}% "
                f"| {r['worst_transformed_catch@thr'] * 100:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="random")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only a seeded random subsample of N")
    ap.add_argument("--limit-seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--conditions", default="",
                    help="comma list of grid conditions to run (must include clean); default all 15")
    ap.add_argument("--threshold", type=float, default=None,
                    help="FIXED product cut-off; every @thr number is read at this line. "
                         "Omitted -> legacy Youden pick on this set's clean scores (comparison only).")
    args = ap.parse_args(argv)

    samples = load_manifest(args.manifest)
    if args.limit:
        # seeded random subsample — never head-truncate (a class-sorted manifest
        # would otherwise yield a single-class subset -> NaN AUROC)
        import random
        random.Random(args.limit_seed).shuffle(samples)
        samples = samples[: args.limit]
    model = load_model(args.model)

    results, summary, errors, per_generator, scores_dump = evaluate(
        model, samples, topk=args.topk, threshold=args.threshold,
        conditions=[c.strip() for c in args.conditions.split(",") if c.strip()] or None)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"model": model.name, "summary": summary, "conditions": results,
                   "per_generator": per_generator}, f, indent=1)
    with open(os.path.join(args.out, "robustness_table.md"), "w") as f:
        f.write(to_markdown(results, summary, model.name, per_generator))
    with open(os.path.join(args.out, "errors_clean.json"), "w") as f:
        json.dump(errors, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "scores.npz"), **scores_dump)  # re-read at any cut-off

    print(to_markdown(results, summary, model.name, per_generator))
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()

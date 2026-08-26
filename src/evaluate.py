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


def evaluate(model, samples, topk=20):
    y = np.asarray([s.label for s in samples])
    results, all_scores = {}, {}

    for name, tf in EVAL_GRID:
        all_scores[name] = _score_condition(model, samples, tf)

    thr = pick_threshold(y, all_scores["clean"])
    for name, _ in EVAL_GRID:
        results[name] = condition_report(y, all_scores[name], thr)

    aurocs = [results[n]["auroc"] for n, _ in EVAL_GRID if n != "clean"]
    summary = {
        "threshold_frozen_on_clean": thr,
        "clean_auroc": results["clean"]["auroc"],
        "mean_transformed_auroc": float(np.nanmean(aurocs)),
        "worst_transformed_auroc": float(np.nanmin(aurocs)),
        "worst_condition": min(
            (n for n, _ in EVAL_GRID if n != "clean"), key=lambda n: results[n]["auroc"]
        ),
    }

    # error analysis on clean: most confident mistakes at the frozen threshold
    s = all_scores["clean"]
    fp = [(samples[i].path, float(s[i])) for i in np.argsort(-s) if y[i] == 0 and s[i] >= thr][:topk]
    fn = [(samples[i].path, float(s[i])) for i in np.argsort(s) if y[i] == 1 and s[i] < thr][:topk]
    errors = {"false_positives": fp, "false_negatives": fn}

    return results, summary, errors


def to_markdown(results, summary, model_name):
    lines = [
        f"# Robustness evaluation — model `{model_name}`",
        "",
        f"Threshold frozen on clean val: {summary['threshold_frozen_on_clean']:.4f}",
        "",
        "| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['auroc']:.4f} | {r['balanced_acc@frozen_thr']:.4f} "
            f"| {r['fpr@tpr95']:.4f} | {r['n']} |"
        )
    lines += [
        "",
        f"**Clean AUROC:** {summary['clean_auroc']:.4f} · "
        f"**Mean transformed:** {summary['mean_transformed_auroc']:.4f} · "
        f"**Worst:** {summary['worst_transformed_auroc']:.4f} ({summary['worst_condition']})",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="random")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N samples")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args(argv)

    samples = load_manifest(args.manifest)
    if args.limit:
        samples = samples[: args.limit]
    model = load_model(args.model)

    results, summary, errors = evaluate(model, samples, topk=args.topk)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"model": model.name, "summary": summary, "conditions": results}, f, indent=1)
    with open(os.path.join(args.out, "robustness_table.md"), "w") as f:
        f.write(to_markdown(results, summary, model.name))
    with open(os.path.join(args.out, "errors_clean.json"), "w") as f:
        json.dump(errors, f, indent=1)

    print(to_markdown(results, summary, model.name))
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()

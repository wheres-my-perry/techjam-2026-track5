"""ROBUSTNESS UNDER STACKED AUGMENTATION — models across, augmentation depth down.

Thinh, 2026-09-01: "each column corresponds to a model; each row corresponds to number of times the
image has been augmented; the i-th row should be (i-1) time augmented."

Depth k = k DISTINCT transform families from the contest grid composed on the SAME image (drawn
without replacement, so depth 6 = every family exactly once). Distinct rather than repeated,
because repeating one family mostly compounds a single artefact -- JPEG twice is just harsher JPEG
-- while distinct families are what a real repost chain does to an image.

Recall is read at ONE cut-off per model, fixed at 1% false alarms on THAT model's CLEAN reals, so
the curve shows degradation and not a threshold sliding underneath it. The false-alarm block is
printed too: a model can hold its recall simply by drifting every score upward.

    python -m scripts.depth_ladder canon6_mlp canon6_mlp2_a1 ...   [--md out.md]
reads outputs/pe_ft/depth_<name>/scores.npz
"""
from __future__ import annotations
import argparse, os, numpy as np
from sklearn.metrics import roc_auc_score

DEPTHS = [("0 (clean)", "clean"), ("1", "stack1_rand"), ("2", "stack2_rand"), ("3", "stack3_rand"),
          ("4", "stack4_rand"), ("5", "stack5_rand"), ("6 (all)", "stack6_rand")]
SHORT = {"canon6": "linear", "canon6_mlp": "MLP (shipped)", "canon6_mlp_consist": "A (trunk)",
         "canon6_mlp_consist_lowlr": "A+lowLR", "canon6_mlp2_a1": "B (a=1.0)",
         "canon6_mlp2_a6": "B (a=6)", "canon6_tail": "C (tail)",
         "canon6_tail_a3": "C+sim", "canon6pe_mlp": "canon6_mlp+edits"}


def load(name):
    f = f"outputs/pe_ft/depth_{name}/scores.npz"
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def block(title, models, data, fn, w, note=""):
    out = [f"\n{title}"]
    if note:
        out.append(note)
    hdr = f"  {'augmentations':<14s}" + "".join(f"{SHORT.get(m, m)[:w-1]:>{w}s}" for m in models)
    out += [hdr, "  " + "-" * (len(hdr) - 2)]
    for label, cond in DEPTHS:
        row = f"  {label:<14s}"
        for m in models:
            v = fn(data[m], cond) if data.get(m) is not None else None
            row += f"{'--':>{w}s}" if v is None else f"{v:>{w}}"
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    data = {m: load(m) for m in a.names}
    models = [m for m in a.names if data[m] is not None]
    if not models:
        print("no depth_* scores found"); return
    thr = {m: float(np.quantile(data[m]["score_clean"][data[m]["labels"] == 0], 1 - a.fa))
           for m in models}

    def recall(o, c):
        k = f"score_{c}"
        if k not in o.files: return None
        y = o["labels"]; m = [n for n in models if data[n] is o][0]
        return f"{(o[k][y == 1] >= thr[m]).mean() * 100:.1f}%"

    def auroc(o, c):
        k = f"score_{c}"
        if k not in o.files: return None
        y = o["labels"]
        return f"{roc_auc_score(y, o[k]):.4f}" if len(set(y)) == 2 else None

    def fa(o, c):
        k = f"score_{c}"
        if k not in o.files: return None
        y = o["labels"]; m = [n for n in models if data[n] is o][0]
        return f"{(o[k][y == 0] >= thr[m]).mean() * 100:.1f}%"

    lines = ["ROBUSTNESS UNDER STACKED AUGMENTATION",
             "Row k = k DISTINCT transform families composed on the same image (6 = all of them).",
             f"n = {len(data[models[0]]['labels'])} judges' images per cell.",
             "",
             "  cut-off per model (1% false alarms on that model's CLEAN reals):"]
    for m in models:
        lines.append(f"    {SHORT.get(m, m):<18s} {thr[m]:.4f}")
    lines += block("RECALL — AI images caught", models, data, recall, 13)
    # total drop, clean -> all six
    drop = "  " + f"{'drop 0->6':<14s}"
    for m in models:
        o = data[m]; y = o["labels"]
        if "score_stack6_rand" in o.files:
            d = ((o["score_clean"][y == 1] >= thr[m]).mean()
                 - (o["score_stack6_rand"][y == 1] >= thr[m]).mean()) * 100
            drop += f"{-d:>12.1f} "
        else:
            drop += f"{'--':>13s}"
    lines += ["  " + "-" * 60, drop]
    lines += block("AUROC", models, data, auroc, 13)
    lines += block("FALSE ALARMS — real photos wrongly flagged", models, data, fa, 13,
                   "  (a model can hold recall by drifting every score up; this is the check)")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        with open(a.md, "w") as f:
            f.write("# Robustness under stacked augmentation\n\n```\n" + txt + "\n```\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()

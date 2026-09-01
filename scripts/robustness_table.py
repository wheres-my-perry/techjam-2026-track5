"""Deliverable 4: the compact clean-vs-transformed robustness summary.

The brief asks for "a compact table or visual summary comparing performance on
clean images versus transformed images". Compact is a requirement, so the 15
single-transform conditions are grouped into the SIX transform families the brief
itself tabulates, each shown as mean (worst) over its parameter settings. Clean is
the baseline row every other row is read against.

Two metrics, deliberately: AUROC is threshold-free and shows ranking quality, while
caught/flagged at the shipped cut-off shows what a user actually experiences. The
brief asks for discussion of false positives, and an AUROC-only table cannot show them.

    python -m scripts.robustness_table --npz outputs/pe_ft/eval_canon6_test/scores.npz \
        --label "canon6 held-out test" --fa 0.01 --md docs/figures/robustness_canon6.md
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

# the brief's six families, in the brief's own order
FAMILIES = [
    ("JPEG compression", "quality 90/70/50/30", ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"]),
    ("Gaussian blur",    "sigma 0.5/1.0/2.0",   ["blur_s0.5", "blur_s1.0", "blur_s2.0"]),
    ("Resize -> upscale", "scale 0.5x / 0.25x", ["resize_0.5x", "resize_0.25x"]),
    ("Gaussian noise",   "sigma .02/.05/.10",   ["noise_s0.02", "noise_s0.05", "noise_s0.10"]),
    ("Colour jitter",    "b/c/s +20% together (implemented cell)", ["jitter_20"]),
    ("Centre crop",      "80%",                 ["crop_80"]),
]
STACKS = [
    ("2 transforms", ["stack2_rand"]), ("3 transforms", ["stack3_rand"]),
    ("4 transforms", ["stack4_rand"]), ("5 transforms", ["stack5_rand"]),
    ("6 transforms", ["stack6_rand"]),
]
CHAINS = [
    ("repost chain", ["chain_repost"]), ("JPEG twice", ["jpeg_twice"]),
    ("blur+JPEG", ["blur1_jpeg70"]), ("noise+JPEG", ["noise05_jpeg70"]),
    ("crop+resize", ["crop80_resize05"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default="test set")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=None,
                    help="use this cut-off instead of deriving one at --fa")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    o = np.load(a.npz)
    y = o["labels"]
    have = {k[len("score_"):] for k in o.files if k.startswith("score_")}
    thr = a.threshold if a.threshold is not None else \
        float(np.quantile(o["score_clean"][y == 0], 1 - a.fa))

    def stat(conds):
        conds = [c for c in conds if c in have]
        if not conds:
            return None
        aucs = [roc_auc_score(y, o[f"score_{c}"]) for c in conds]
        caught = [float((o[f"score_{c}"][y == 1] >= thr).mean()) for c in conds]
        flagged = [float((o[f"score_{c}"][y == 0] >= thr).mean()) for c in conds]
        return aucs, caught, flagged

    def fmt(s, single=False):
        aucs, caught, flagged = s
        if single or len(aucs) == 1:
            return f"{aucs[0]:.4f} | {caught[0]*100:.1f}% | {flagged[0]*100:.1f}%"
        return (f"{np.mean(aucs):.4f} ({min(aucs):.4f}) | "
                f"{np.mean(caught)*100:.1f}% ({min(caught)*100:.1f}%) | "
                f"{np.mean(flagged)*100:.1f}% ({max(flagged)*100:.1f}%)")

    n_r, n_f = int((y == 0).sum()), int((y == 1).sum())
    L = [f"### Robustness — {a.label}", "",
         f"n = {n_r + n_f} ({n_r} real / {n_f} AI). Cut-off **{thr:.4f}**, chosen at "
         f"{a.fa*100:.3g}% false alarms on clean reals" + (" (fixed)" if a.threshold else "") + ".",
         "Cells are mean (worst) over the parameter settings in that row. "
         "*Caught* = AI images at or above the cut-off; *flagged* = real images at or above it.", "",
         "| Condition | Parameters | AUROC | Caught | Flagged |", "|---|---|---|---|---|"]

    c = stat(["clean"])
    L.append(f"| **Clean (baseline)** | — | {fmt(c, True)} |")
    for name, params, conds in FAMILIES:
        s = stat(conds)
        if s:
            L.append(f"| {name} | {params} | {fmt(s)} |")
    if any(c in have for _, cs in STACKS for c in cs):
        L.append("| *— stacked (random subset of the six families) —* | | | | |")
        for name, conds in STACKS:
            s = stat(conds)
            if s:
                L.append(f"| {name} | random | {fmt(s, True)} |")
    if any(c in have for _, cs in CHAINS for c in cs):
        L.append("| *— fixed real-world chains —* | | | | |")
        for name, conds in CHAINS:
            s = stat(conds)
            if s:
                L.append(f"| {name} | fixed | {fmt(s, True)} |")

    allt = [c for c in have if c != "clean"]
    st = stat(allt)
    L += ["", f"**All {len(allt)} transformed conditions:** AUROC mean {np.mean(st[0]):.4f}, "
              f"worst {min(st[0]):.4f} · caught mean {np.mean(st[1])*100:.1f}% · "
              f"flagged mean {np.mean(st[2])*100:.1f}%, worst {max(st[2])*100:.1f}%."]

    out = "\n".join(L)
    print(out)
    if a.md:
        import os
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write(out + "\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()

"""Concrete confusion matrix at ONE fixed threshold, overall and per dimension.

Thinh (2026-08-31): "AUROC is so easy to lie, over a dimension it is easier to have a good
ordering, and thus have a good AUROC; but when you mix them together the order can shuffle
heavily, and that's what AUROC per dimension is hiding."

AUROC only measures ORDERING WITHIN the set it is computed on. Slice by size bucket, generator or
transform and each slice can order almost perfectly while sitting at a different SCORE LEVEL;
pooled under one threshold, reals from a high-scoring slice outrank fakes from a low-scoring one.
Measured on omni: per-bucket AUROC 0.968-0.9997 but per-bucket optimal cut-offs 0.257-0.711, and
at the global cut-off the <=341 bucket flagged 3.3% of reals -- 3x the target. Only the counts
showed it.

Reports, at ONE global cut-off chosen on ALL reals: TP / FP / TN / FN as COUNTS, precision, recall,
FPR -- overall and for every slice, with each slice's own optimal cut-off beside it.

    python -m scripts.confusion --npz outputs/pe_ft/eval_omni_benchmark/scores.npz \
        --manifest data/manifests/benchmark.csv --fa 0.01
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ORDER = ["<=341", "342-512", "513-768", "769-1024", ">1024"]


def _p(path):
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def buck(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def matrix(y, s, thr):
    tp = int(((s >= thr) & (y == 1)).sum()); fn = int(((s < thr) & (y == 1)).sum())
    fp = int(((s >= thr) & (y == 0)).sum()); tn = int(((s < thr) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return tp, fp, tn, fn, prec, rec, fpr


def report(tag, y, s, thr, extra=""):
    tp, fp, tn, fn, prec, rec, fpr = matrix(y, s, thr)
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
    own = float(np.quantile(s[y == 0], 0.99)) if (y == 0).any() else float("nan")
    print(f"  {tag:26s} {len(y):7d} {auc:8.4f} | {tp:7d} {fp:6d} {tn:7d} {fn:6d} | "
          f"{prec*100:7.1f}% {rec*100:7.1f}% {fpr*100:6.1f}% | {own:8.4f} {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", default=None, help="for the `long` column (size buckets)")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=None, help="use a fixed cut-off instead")
    ap.add_argument("--condition", default="clean")
    a = ap.parse_args()

    o = np.load(a.npz, allow_pickle=True)
    y = o["labels"]
    key = f"score_{a.condition}"
    if key not in o.files:
        raise SystemExit(f"{a.condition} not in npz")
    s = o[key]
    gens = [str(g) for g in o["generators"]] if "generators" in o.files else [""] * len(y)
    paths = [_p(p) for p in o["paths"]] if "paths" in o.files else [""] * len(y)

    thr = a.threshold if a.threshold is not None else float(np.quantile(s[y == 0], 1 - a.fa))
    auc = roc_auc_score(y, s)
    tp, fp, tn, fn, prec, rec, fpr = matrix(y, s, thr)

    print(f"{a.npz}   condition={a.condition}")
    print(f"\nPOOLED  AUROC {auc:.4f}   n={len(y)} ({int((y==0).sum())} real / {int((y==1).sum())} AI)")
    print(f"ONE GLOBAL CUT-OFF = {thr:.4f}"
          + ("  (given)" if a.threshold is not None
             else f"  (at {a.fa*100:.3g}% false alarms on all reals)"))
    print("\nCONFUSION MATRIX at that cut-off — counts, not rates")
    print(f"                       predicted AI    predicted real")
    print(f"  actually AI   {tp:14d} {fn:17d}")
    print(f"  actually real {fp:14d} {tn:17d}")
    print(f"\n  precision {prec*100:.1f}%   recall {rec*100:.1f}%   "
          f"false-alarm rate {fpr*100:.2f}%   accuracy {(tp+tn)/len(y)*100:.1f}%")
    print(f"  In words: of {int((y==1).sum())} AI images {tp} are caught and {fn} slip through; "
          f"of {int((y==0).sum())} real photos {fp} are wrongly flagged.")

    hdr = (f"\n  {'slice':26s} {'n':>7s} {'AUROC':>8s} | {'TP':>7s} {'FP':>6s} {'TN':>7s} "
           f"{'FN':>6s} | {'prec':>8s} {'recall':>8s} {'FPR':>6s} | {'own thr':>8s}")
    sep = "  " + "-" * 108

    if a.manifest and os.path.exists(a.manifest):
        long_of = {}
        for r in csv.DictReader(open(a.manifest, newline="")):
            if r.get("long"):
                long_of[_p(r["path"])] = int(r["long"])
        idx = defaultdict(list)
        for i, p in enumerate(paths):
            L = long_of.get(p)
            if L:
                idx[buck(L)].append(i)
        if idx:
            print("\nBY NATIVE SIZE BUCKET (all at the SAME global cut-off)")
            print(hdr); print(sep)
            reals = [i for i in range(len(y)) if y[i] == 0]
            for b in ORDER:
                ii = idx.get(b)
                if not ii or len(ii) < 20:
                    continue
                sub = np.array(sorted(set(ii) | set(reals))) if (y[ii] == 0).sum() < 10 else np.array(ii)
                report(b, y[sub], s[sub], thr)

    by_gen = defaultdict(list)
    for i, g in enumerate(gens):
        if y[i] == 1 and g:
            by_gen[g].append(i)
    if by_gen:
        reals = np.array([i for i in range(len(y)) if y[i] == 0])
        print(f"\nBY GENERATOR (each vs ALL {len(reals)} reals, same global cut-off)")
        print(hdr); print(sep)
        rows = []
        for g, ii in by_gen.items():
            sub = np.concatenate([reals, np.array(ii)])
            tp2, _, _, fn2, _, rec2, _ = matrix(y[sub], s[sub], thr)
            rows.append((rec2, g, sub))
        for rec2, g, sub in sorted(rows):
            report(g, y[sub], s[sub], thr)
        print(f"\n  Sorted by recall: the generators at the top are the ones that slip through.")


if __name__ == "__main__":
    main()

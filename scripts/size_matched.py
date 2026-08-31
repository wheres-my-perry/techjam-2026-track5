"""Re-read an evaluation size-matched, per native-size bucket.

Why this is mandatory here: canonicalization makes every image 176x176, so
shortcut_audit (which reads the canonical files) sees constant width/height/format
and is STRUCTURALLY BLIND to native size. A set can pass it at 0.62 while still
having "big => real, small => fake" baked in -- and the shrink-to-320 factor leaves
a physical trace, so that is learnable without looking at content
(docs/LESSONS_FOR_TEAMMATES.md section 2).

Measured on canon_unseen6: buckets 342-512, 513-768 and >1024 hold no fakes at all,
reals run to 7712px native while fakes stop at 1024. A pooled AUROC over that set
would partly be measuring image size. The original unseen-64 set had the same flaw
(metadata-only 0.92) and was reported size-matched for exactly this reason.

Reads scores.npz (which stores `paths`) and joins to the manifest's `long` column,
so no GPU work is repeated.

    python -m scripts.size_matched --npz outputs/pe_ft/eval_canon6_unseen/scores.npz \
        --manifest data/manifests/canon_unseen6.csv [--condition clean]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score



def _p(path):
    """Normalize a manifest path for joining.

    src.data.load_manifest prefixes paths with "./" while the csv stores them bare, so a dict
    keyed on one and looked up with the other silently matches NOTHING -- val_by_bucket printed an
    empty table and size_matched would have reported "no bucket has both classes".
    """
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p

ORDER = ["<=341", "342-512", "513-768", "769-1024", ">1024"]


def buck(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    o = np.load(a.npz, allow_pickle=True)
    key = f"score_{a.condition}"
    if key not in o.files:
        raise SystemExit(f"{a.condition} not in npz ({[k[6:] for k in o.files if k.startswith('score_')][:6]}...)")
    paths = [str(p) for p in o["paths"]]
    y = o["labels"]
    s = o[key]

    long_of = {}
    for r in csv.DictReader(open(a.manifest, newline="")):
        if r.get("long"):
            long_of[_p(r["path"])] = int(r["long"])

    idx_by = defaultdict(lambda: {0: [], 1: []})
    missing = 0
    for i, p in enumerate(paths):
        L = long_of.get(_p(p))
        if L is None:
            missing += 1
            continue
        idx_by[buck(L)][int(y[i])].append(i)

    print(f"{a.npz}  condition={a.condition}")
    if missing:
        print(f"  ({missing} scored rows had no `long` in the manifest and were skipped)")
    print(f"\n  {'bucket':10s} {'real':>7s} {'fake':>7s} {'AUROC':>9s}   note")
    keep = []
    rng = random.Random(a.seed)
    for b in ORDER:
        r, f = idx_by[b][0], idx_by[b][1]
        if not (r or f):
            continue
        if not r or not f:
            print(f"  {b:10s} {len(r):7d} {len(f):7d} {'--':>9s}   NOT SCORABLE (one class absent) - dropped")
            continue
        n = min(len(r), len(f))
        rr, ff = r[:], f[:]
        rng.shuffle(rr); rng.shuffle(ff)
        sel = rr[:n] + ff[:n]
        auc = roc_auc_score(y[sel], s[sel])
        keep += sel
        print(f"  {b:10s} {len(r):7d} {len(f):7d} {auc:9.4f}   matched to {n} each")

    if not keep:
        raise SystemExit("\nNo bucket has both classes: this set cannot be scored size-matched.")
    auc = roc_auc_score(y[keep], s[keep])
    nr = int((y[keep] == 0).sum())
    thr = float(np.quantile(s[keep][y[keep] == 0], 1 - a.fa))
    caught = float((s[keep][y[keep] == 1] >= thr).mean())
    print(f"\n  SIZE-MATCHED POOLED: AUROC {auc:.4f}  (n={len(keep)}: {nr} real / {len(keep)-nr} fake)")
    print(f"  at {a.fa*100:.3g}% false alarms (cut-off {thr:.4f}): {caught*100:.1f}% of AI images caught")
    unmatched = roc_auc_score(y, s)
    print(f"\n  (unmatched pooled AUROC over all {len(y)} rows: {unmatched:.4f} — "
          f"{'INFLATED by size' if unmatched > auc else 'not inflated'} by {unmatched-auc:+.4f})")


if __name__ == "__main__":
    main()

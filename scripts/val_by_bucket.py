"""Validation metrics per native-size bucket — does any bucket need more training data?

Thinh (2026-08-31): "I want to see val metrics per bucket, to see if we need more training data
from any distribution."

Each native-size bucket is a different forensic regime: canonicalize --long 320 only shrinks images
ABOVE 320, so <=341 reaches the model un-rescaled while 769-1024 is downscaled 2.4-3.2x. A bucket
that scores worse than the others is the one starved of training data (or intrinsically harder),
and is where more data should go.

Prints, per bucket: n, AUROC, and recall/false-alarm at one shared cut-off, alongside how many
TRAINING images that bucket had -- so the two can be read together.

    python -m scripts.val_by_bucket --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt" \
        --manifest data/manifests/canon6_val.csv --train data/manifests/canon6_train.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import load_image, load_manifest
from src.model import load_model



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
BATCH = 32


def buck(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="data/manifests/canon6_val.csv")
    ap.add_argument("--train", default="data/manifests/canon6_train.csv")
    ap.add_argument("--per-bucket", type=int, default=1200, help="images scored per bucket")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    long_of, bucket_of = {}, {}
    for r in csv.DictReader(open(a.manifest, newline="")):
        if r.get("long"):
            bucket_of[_p(r["path"])] = buck(r["long"])
    train_n = Counter()
    for r in csv.DictReader(open(a.train, newline="")):
        if r.get("long"):
            train_n[buck(r["long"])] += 1

    samples = load_manifest(a.manifest)
    by = defaultdict(list)
    for s in samples:
        b = bucket_of.get(_p(s.path))
        if b:
            by[b].append(s)

    model = load_model(a.model)
    rng = random.Random(a.seed)
    print(f"model: {a.model}\nval: {a.manifest}\n")
    print(f"{'bucket':10s} {'train imgs':>11s} {'val n':>7s} {'AUROC':>8s} "
          f"{'cut-off':>8s} {'caught':>8s} {'flagged':>8s}")
    print("-" * 70)
    rows = []
    for b in ORDER:
        items = by.get(b, [])
        if len(items) < 40:
            continue
        real = [s for s in items if s.label == 0]
        fake = [s for s in items if s.label == 1]
        k = min(len(real), len(fake), a.per_bucket // 2)
        rng.shuffle(real); rng.shuffle(fake)
        sel = real[:k] + fake[:k]
        y = np.array([s.label for s in sel])
        scores = []
        for i in range(0, len(sel), BATCH):
            scores.extend(model.predict([load_image(s.path) for s in sel[i:i + BATCH]]))
        scores = np.asarray(scores)
        auc = roc_auc_score(y, scores)
        thr = float(np.quantile(scores[y == 0], 1 - a.fa))
        caught = float((scores[y == 1] >= thr).mean())
        flagged = float((scores[y == 0] >= thr).mean())
        rows.append((b, auc, caught))
        print(f"{b:10s} {train_n[b]:11,d} {len(sel):7d} {auc:8.4f} {thr:8.4f} "
              f"{caught*100:7.1f}% {flagged*100:7.1f}%")

    if len(rows) > 1:
        worst = min(rows, key=lambda r: r[1])
        best = max(rows, key=lambda r: r[1])
        print(f"\n  best  {best[0]} {best[1]:.4f}   worst {worst[0]} {worst[1]:.4f}   "
              f"spread {best[1]-worst[1]:.4f}")
        print(f"  Cut-off is chosen per bucket at {a.fa*100:.3g}% false alarms, so buckets are "
              f"compared at a matched operating point.")
        print(f"  If the worst bucket is also the one with the fewest training images, more data "
              f"there should help; if not, it is intrinsically harder and more data will not fix it.")


if __name__ == "__main__":
    main()

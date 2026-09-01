"""Validation metrics per native-size bucket — does any bucket need more training data?

Thinh (2026-08-31): "I want to see val metrics per bucket, to see if we need more training data
from any distribution."

Each native-size bucket is a different forensic regime: canonicalize --long 320 only shrinks images
ABOVE 320, so <=341 reaches the model un-rescaled while 769-1024 is downscaled 2.4-3.2x. A bucket
that scores worse than the others is the one starved of training data (or intrinsically harder),
and is where more data should go.

Prints, per bucket: n, AUROC, and recall/false-alarm at one shared cut-off, alongside how many
TRAINING images that bucket had -- so the two can be read together.

    python -m scripts.val_by_bucket --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt" \
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
    rows, pooled_y, pooled_s, pooled_b = [], [], [], []
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
        rows.append({"bucket": b, "auc": auc, "own_thr": thr, "y": y, "s": scores,
                     "own_caught": float((scores[y == 1] >= thr).mean()),
                     "own_flagged": float((scores[y == 0] >= thr).mean())})
        pooled_y.append(y); pooled_s.append(scores); pooled_b += [b] * len(y)

    if not rows:
        raise SystemExit("no bucket had enough rows to score")

    Y = np.concatenate(pooled_y); S = np.concatenate(pooled_s)
    pooled_auc = roc_auc_score(Y, S)
    GLOBAL = float(np.quantile(S[Y == 0], 1 - a.fa))

    # A per-bucket cut-off is not what ships. Production applies ONE threshold to every image
    # regardless of its size, so the honest per-bucket figures are the ones read at that single
    # global cut-off (Thinh 2026-08-31). Both are printed, and the gap between them is the cost
    # of having one threshold.
    print(f"POOLED OVER THE WHOLE SET   AUROC {pooled_auc:.4f}   n={len(Y)} "
          f"({int((Y==0).sum())} real / {int((Y==1).sum())} fake)")
    print(f"ONE GLOBAL CUT-OFF at {a.fa*100:.3g}% false alarms on ALL reals = {GLOBAL:.4f}  "
          f"<- this is what a deployed product uses")
    g_caught = float((S[Y == 1] >= GLOBAL).mean()); g_flag = float((S[Y == 0] >= GLOBAL).mean())
    print(f"  at that cut-off, over the whole set: {g_caught*100:.1f}% of AI caught, "
          f"{g_flag*100:.1f}% of reals flagged\n")

    print(f"{'bucket':10s} {'train':>8s} {'n':>6s} {'AUROC':>8s} | "
          f"{'AT THE GLOBAL CUT-OFF':>26s} | {'(at its own cut-off)':>24s}")
    print(f"{'':10s} {'':8s} {'':6s} {'':8s} | {'caught':>12s} {'flagged':>12s} | "
          f"{'cut-off':>10s} {'caught':>12s}")
    print("-" * 104)
    for r in rows:
        y, sc = r["y"], r["s"]
        gc = float((sc[y == 1] >= GLOBAL).mean()); gf = float((sc[y == 0] >= GLOBAL).mean())
        print(f"{r['bucket']:10s} {train_n[r['bucket']]:8,d} {len(y):6d} {r['auc']:8.4f} | "
              f"{gc*100:11.1f}% {gf*100:11.1f}% | {r['own_thr']:10.4f} {r['own_caught']*100:11.1f}%")

    worst = min(rows, key=lambda r: r["auc"]); best = max(rows, key=lambda r: r["auc"])
    print(f"\n  best {best['bucket']} {best['auc']:.4f}   worst {worst['bucket']} "
          f"{worst['auc']:.4f}   spread {best['auc']-worst['auc']:.4f}")
    spread = max(r["own_thr"] for r in rows) - min(r["own_thr"] for r in rows)
    print(f"  per-bucket optimal cut-offs span {spread:.3f} "
          f"({min(r['own_thr'] for r in rows):.3f}-{max(r['own_thr'] for r in rows):.3f}); "
          f"that spread is exactly what one fixed threshold has to absorb.")
    print(f"  If the worst bucket is also the one with the fewest training images, more data there "
          f"should help; if not, it is intrinsically harder and more data will not fix it.")


if __name__ == "__main__":
    main()

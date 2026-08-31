"""Build a TRAINING corpus from OmniFake's val split (90K real + 90K fake, 45 generators).

Thinh's design (2026-08-31): train on OmniFake only, and benchmark on everything OmniFake does NOT
cover. That decouples the two pipelines completely — the corpus that builds the model and the
corpus that measures it then share no source, no preprocessing and no construction, so a good
benchmark number cannot come from our own pipeline.

Why OmniFake's val split rather than its train split: the train split's real half is a 41-part,
217 GB archive that will not fit here, while val is 17 parts / 88 GB and already holds 90K reals
matched to 90K fakes over all 45 generators — enough to train on. We are not comparing against the
paper's numbers, so using their val split as our training data costs nothing.

Layout produced by the archive:  val/<Generator>/**  and  val/real/**

    python -m scripts.build_omnifake_train --root data/omnival/data/val \
        --out data/manifests/raw_omnitrain.csv --cap-per-generator 2600 --cap-real 100000
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIRS = {"real", "reals", "0_real", "nature"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/omnival/data/val")
    ap.add_argument("--out", default="data/manifests/raw_omnitrain.csv")
    ap.add_argument("--cap-per-generator", type=int, default=2600)
    ap.add_argument("--cap-real", type=int, default=100000)
    a = ap.parse_args()

    rows, per, sizes = [], Counter(), Counter()
    for d in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, d)
        if not os.path.isdir(p):
            continue
        is_real = d.lower() in REAL_DIRS
        cap = a.cap_real if is_real else a.cap_per_generator
        n = 0
        for dp, _, fns in os.walk(p):
            for fn in sorted(fns):
                if n >= cap:
                    break
                if os.path.splitext(fn)[1].lower() not in IMG:
                    continue
                fp = os.path.join(dp, fn)
                try:
                    with Image.open(fp) as im:
                        w, h = im.size
                except Exception:
                    continue
                rows.append({"path": fp, "label": 0 if is_real else 1,
                             "generator": "" if is_real else "omni_" + d.lower(),
                             "source": "omnifake_real" if is_real else "omnifake",
                             "w": w, "h": h})
                sizes[max(w, h)] += 1
                n += 1
            if n >= cap:
                break
        per[d] = n
        print(f"  {d:26s} {n:6d} {'REAL' if is_real else 'fake'}", flush=True)

    empty = [d for d, n in per.items() if n == 0]
    if empty:
        print(f"!! produced NO images: {empty}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source", "w", "h"])
        w_.writeheader(); w_.writerows(rows)
    nr = sum(1 for r in rows if r["label"] == 0)
    ngen = sum(1 for d, n in per.items() if n and d.lower() not in REAL_DIRS)
    print(f"\n{len(rows)} rows ({nr} real / {len(rows)-nr} fake) from {ngen} generators -> {a.out}")
    print("native long side (top):", sizes.most_common(8))


if __name__ == "__main__":
    main()

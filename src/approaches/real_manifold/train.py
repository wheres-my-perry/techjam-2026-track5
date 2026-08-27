"""Fit the real-manifold model on REAL images only (label==0 rows of the manifest).

    python -m src.approaches.real_manifold.train --train data/manifests/wildfake_train.csv \
        [--augment-views 2] [--limit 8000] --out outputs/real_manifold/baseline.npz

--augment-views N: also include N contest-transformed copies of each real, so
"real" is learned to INCLUDE compressed/blurred/resized reals (the JPEG-hazard
mitigation from docs/approaches/02-real-manifold.md). CPU-only; no GPU needed.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from src.data import load_image, load_manifest
from src.metrics import auroc
from src.transforms import random_train_transform
from .model import RealManifoldModel, features, fit, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="", help="optional: manifest WITH fakes, "
                    "reports val AUROC after fitting")
    ap.add_argument("--augment-views", type=int, default=2)
    ap.add_argument("--limit", type=int, default=8000, help="max real images used")
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/real_manifold/baseline.npz")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reals = [s for s in load_manifest(args.train) if s.label == 0]
    rng.shuffle(reals)
    reals = reals[: args.limit]
    print(f"fitting on {len(reals)} reals x (1 clean + {args.augment_views} augmented views)")

    rows = []
    for n, s in enumerate(reals):
        img = load_image(s.path)
        rows.append(features(img))
        for _ in range(args.augment_views):
            rows.append(features(random_train_transform(img, rng)))
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(reals)} reals featurized", flush=True)
    X = np.vstack(rows)

    params = fit(X)
    save(params, args.out, meta={"n_reals": len(reals),
                                 "augment_views": args.augment_views,
                                 "seed": args.seed})
    print(f"saved -> {args.out} ({X.shape[0]} feature rows, {X.shape[1]} dims)")

    if args.val:
        val = load_manifest(args.val)
        rng.shuffle(val)
        val = val[: args.val_limit]
        model = RealManifoldModel(args.out)
        ys, ss = [], []
        for n, s in enumerate(val):
            ss.append(float(model.predict([load_image(s.path)])[0]))
            ys.append(s.label)
            if (n + 1) % 500 == 0:
                print(f"  val {n + 1}/{len(val)}", flush=True)
        print(f"val AUROC (never saw a fake in training): {auroc(ys, ss):.4f}")


if __name__ == "__main__":
    main()

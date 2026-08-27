"""Train the spectral detector: extract features, fit logistic regression.

    python -m src.approaches.spectral.train --train data/manifests/wildfake_train.csv \
        --val data/manifests/wildfake_val.csv --limit 20000 \
        --out outputs/spectral/baseline.npz

CPU-only. --limit N takes a seeded random subsample of the train manifest
(balanced by whatever the shuffle gives; manifests are large so this bounds
extraction time). Progress prints every 500 images. Feature extraction is the
whole cost; the fit takes seconds.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

from src.data import load_image, load_manifest
from src.metrics import auroc
from .model import features


def extract(samples, tag: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            X.append(features(load_image(s.path)))
            y.append(s.label)
        except Exception as e:  # unreadable file: skip, keep going
            print(f"skip {s.path}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(samples) - i - 1) / max(rate, 1e-9)
            print(f"{tag}: {i+1}/{len(samples)} ({rate:.1f}/s, eta {eta/60:.0f}m)",
                  flush=True)
    return np.asarray(X), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--val-limit", type=int, default=3000)
    ap.add_argument("--out", default="outputs/spectral/baseline.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = load_manifest(args.train)
    rng.shuffle(train)
    train = train[: args.limit]
    val = load_manifest(args.val)
    rng.shuffle(val)
    val = val[: args.val_limit]
    print(f"train {len(train)} val {len(val)}", flush=True)

    Xtr, ytr = extract(train, "train")
    f_mean = Xtr.mean(axis=0)
    f_std = Xtr.std(axis=0) + 1e-8
    Ztr = (Xtr - f_mean) / f_std

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Ztr, ytr)
    print(f"train fit done; train AUROC "
          f"{auroc(ytr, clf.predict_proba(Ztr)[:, 1]):.4f}", flush=True)

    Xv, yv = extract(val, "val")
    Zv = (Xv - f_mean) / f_std
    val_auc = auroc(yv, clf.predict_proba(Zv)[:, 1])
    print(f"val AUROC {val_auc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, w=clf.coef_[0], b=clf.intercept_[0],
                        f_mean=f_mean, f_std=f_std)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

"""Shortcut-ceiling audit: can trivial metadata alone classify a manifest?

    python -m scripts.shortcut_audit --manifest data/manifests/official_v2.csv

Model-free benchmark validation (standing rule, 2026-08-28): a logistic
regression sees ONLY width, height, min/max side, aspect ratio, file size,
and extension — never a pixel. 5-fold cross-validated AUROC:

    ~0.5        benchmark is clean of metadata shortcuts
    0.5-0.65    mild leak: report results with a caveat, prefer fixing
    > 0.65      FAIL: no model result from this manifest may be reported

official_val scored ~1.0 here (200x200 reals vs 1024 fakes) — this audit
exists so that can never happen silently again.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image

from src.data import load_manifest
from src.metrics import auroc


def features(path: str) -> list[float]:
    with Image.open(os.path.expandvars(path)) as im:  # header only, no decode
        w, h = im.size
        fmt = im.format or "?"
    size = os.path.getsize(os.path.expandvars(path))
    exts = ["JPEG", "PNG", "WEBP"]
    return [w, h, min(w, h), max(w, h), w / h, np.log1p(size)] + \
        [1.0 if fmt == e else 0.0 for e in exts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()
    s = load_manifest(args.manifest)
    random.Random(0).shuffle(s)
    s = s[: args.limit]
    X, y = [], []
    for x in s:
        try:
            X.append(features(x.path))
            y.append(x.label)
        except Exception as e:
            print(f"skip {x.path}: {e}", flush=True)
    X = np.asarray(X)
    y = np.asarray(y)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    a = auroc(y, p)
    verdict = ("CLEAN" if a < 0.55 else
               "MILD LEAK — caveat results" if a <= 0.65 else
               "FAIL — do not report model results from this manifest")
    print(f"metadata-only AUROC: {a:.4f}  [{verdict}]  "
          f"({len(y)} rows, {args.manifest})")


if __name__ == "__main__":
    main()

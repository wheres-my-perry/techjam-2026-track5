"""Pre-download metadata probe for Hugging Face datasets (streaming).

    python -m scripts.remote_probe --dataset bitmind/ArtiFact --samples 800
    python -m scripts.remote_probe --dataset Yejy53/Echo-4o-Image --samples 300

Verifies a candidate dataset's ACTUAL image-size distribution before any bulk
download is approved (rule: never trust the dataset card — WildFake's card
never mentioned its 200x200 thumbnails). Streams N samples over the wire
(cost ~ N x avg image size, i.e. MBs, not the full set), then reports:
- per-label size distribution (top sizes per class), and
- if labels are binary, the metadata-only AUROC (width/height/aspect ->
  logistic regression), same CLEAN / MILD LEAK / FAIL scale as shortcut_audit.

If the dataset's column names differ, the script prints the available
features and exits — rerun with --image-field/--label-field set accordingly.
For local manifests use scripts/shortcut_audit.py instead.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--image-field", default="image")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split=args.split,
                      streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)

    first = next(iter(ds.take(1)))
    if args.image_field not in first or args.label_field not in first:
        print(f"fields available: {sorted(first.keys())}")
        print("rerun with --image-field / --label-field from the list above")
        return

    groups: dict[str, Counter] = {}
    rows, labels = [], []
    n = 0
    for ex in ds.take(args.samples):
        img = ex[args.image_field]
        try:
            w, h = img.size
        except Exception:
            continue
        lab = str(ex[args.label_field])
        groups.setdefault(lab, Counter())[(w, h)] += 1
        rows.append([w, h, min(w, h), max(w, h), w / h])
        labels.append(lab)
        n += 1
        if n % 200 == 0:
            print(f"  probed {n}/{args.samples}", flush=True)

    print(f"\n== {args.dataset} [{args.split}] — {n} samples probed ==")
    for k in sorted(groups):
        top = ", ".join(f"{w}x{h}:{c}" for (w, h), c in groups[k].most_common(4))
        print(f"label={k:24s} n={sum(groups[k].values()):5d}  {top}")

    uniq = sorted(set(labels))
    if len(uniq) == 2:
        X = np.asarray(rows)
        y = np.asarray([uniq.index(l) for l in labels])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from src.metrics import auroc
        p = cross_val_predict(LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"),
                              X, y, cv=5, method="predict_proba")[:, 1]
        a = auroc(y, p)
        verdict = ("CLEAN" if a < 0.55 else
                   "MILD LEAK — caveat" if a <= 0.65 else
                   "FAIL — size predicts label; do not download for training")
        print(f"\nmetadata-only AUROC (size features): {a:.4f}  [{verdict}]")
    else:
        print(f"\n{len(uniq)} distinct labels — size table above is the "
              "deliverable (binary AUROC needs exactly 2)")


if __name__ == "__main__":
    main()

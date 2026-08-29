"""Score every image in a folder with the current demo model and print a summary.

    python -m scripts.score_dir data/hack/dalle_api --label fake
    python -m scripts.score_dir data/hack/real --label real
    python -m scripts.score_dir some_folder            (no label: scores only)

Uses the same procedure as the app (shrink long side to 320, 27-crop mean).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.data import load_image
from src.model import load_model
from src.predict import iter_image_paths

DEFAULT = "vote(L=320)+pe_ft:outputs/pe_ft/canon3.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--label", choices=["real", "fake"], default=None)
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    paths = list(iter_image_paths(args.folder))
    if not paths:
        raise SystemExit(f"no images under {args.folder}")
    m = load_model(args.model)
    ims, kept = [], []
    for p in paths:
        try:
            ims.append(load_image(p)); kept.append(p)
        except Exception as e:
            print(f"skip {p}: {e}")
    s = m.predict(ims)
    for p, im, v in zip(kept, ims, s):
        print(f"  {v:.3f}  {'AI ' if v >= args.threshold else 'real'}  {str(im.size):13s} {os.path.basename(p)}")
    print(f"\n{len(kept)} images  median {np.median(s):.3f}  mean {s.mean():.3f}  "
          f"share>=0.2 {np.mean(s >= 0.2):.2f}  share>=0.5 {np.mean(s >= 0.5):.2f}")
    if args.label:
        want = args.label == "fake"
        acc = np.mean((s >= args.threshold) == want)
        print(f"labelled {args.label}: correct at threshold {args.threshold}: {acc:.2f}  "
              f"(at 0.5: {np.mean((s >= 0.5) == want):.2f})")


if __name__ == "__main__":
    main()

"""Required contest deliverable: score a directory of images.

Usage:
    python -m src.predict --input <image_dir> --output preds.json [--model SPEC] [--threshold 0.15]

Output JSON: [{"image_path": "...", "pred": 0.87, "label": 1}, ...]
  pred  = P(AI-generated) from the shipped policy (shrink long side to 320, 27-crop grid, mean)
  label = 1 if pred >= threshold (default 0.5)

Choosing the cut-off (measured, pooled over all 15 transform conditions):

    cut-off   judges' set          held-out test        hack set (real files)
              recall / false-alarm recall / false-alarm caught / false alarms
    0.300     99.1% / 4.66%        83.5% / 4.61%        17 of 20 / 0 of 5
    0.500     97.6% / 2.27%        76.8% / 2.24%        17 of 20 / 0 of 5   <- shipped
    0.717     94.8% / 1.01%        68.0% / 0.78%        11 of 20 / 0 of 5

0.717 is the textbook 1%-false-alarm point, but it costs 6 of 20 real-world detections on the hack
set and 9 points of recall on the held-out test, for ~1.2 points of false alarms. 0.5 keeps the
false-alarm rate near 2% while holding recall, so it is the shipped default. Do NOT choose the
cut-off on clean images alone: that gives 0.216, which holds 1.1% false alarms when clean and 22.9%
under JPEG q30, because JPEG shifts every score upward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .data import load_image
from .model import load_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"}
DEFAULT_MODEL = "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt"
DEFAULT_THRESHOLD = 0.5
BATCH = 32


def iter_image_paths(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield os.path.join(dirpath, fn)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of images")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)

    model = load_model(args.model)
    paths = list(iter_image_paths(args.input))
    if not paths:
        print(f"No images found under {args.input}", file=sys.stderr)

    results = []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        images, kept = [], []
        for p in chunk:
            try:
                images.append(load_image(p))
                kept.append(p)
            except Exception as e:  # corrupt file: score 0.5, keep going
                print(f"WARN: failed to load {p}: {e}", file=sys.stderr)
                results.append({"image_path": p, "pred": 0.5, "label": int(0.5 >= args.threshold)})
        if images:
            scores = model.predict(images)
            results.extend(
                {"image_path": p, "pred": round(float(s), 6), "label": int(float(s) >= args.threshold)}
                for p, s in zip(kept, scores)
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {len(results)} predictions -> {args.output}")


if __name__ == "__main__":
    main()

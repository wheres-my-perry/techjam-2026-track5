"""Required contest deliverable: score a directory of images.

Usage:
    python -m src.predict --input <image_dir> --output preds.json [--model random]

Output JSON: [{"image_path": "...", "pred": 0.87}, ...]   pred = P(AI-generated)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .data import load_image
from .model import load_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
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
    ap.add_argument("--model", default="random")
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
                results.append({"image_path": p, "pred": 0.5})
        if images:
            scores = model.predict(images)
            results.extend(
                {"image_path": p, "pred": round(float(s), 6)}
                for p, s in zip(kept, scores)
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {len(results)} predictions -> {args.output}")


if __name__ == "__main__":
    main()

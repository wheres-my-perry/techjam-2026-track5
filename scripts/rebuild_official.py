"""Rebuild the official benchmark with ORIGINAL-resolution COCO val2017 reals.

    python -m scripts.rebuild_official --coco-dir data/coco_orig/val2017

WildFake's coco slice is 200x200 thumbnails while DALL-E fakes are 1024+, so
size alone separated the classes in official_val.csv (2026-08-28 finding).
This keeps the fake rows and swaps the reals for full-resolution COCO files.
"""

from __future__ import annotations

import argparse
import csv
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/manifests/official_val.csv")
    ap.add_argument("--coco-dir", default="data/coco_orig/val2017")
    ap.add_argument("--out", default="data/manifests/official_v2.csv")
    ap.add_argument("--cap", type=int, default=4998)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.old)))
    fakes = [r for r in rows if r["label"] == "1"]
    imgs = sorted(f for f in os.listdir(args.coco_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(args.seed).shuffle(imgs)
    reals = [{"path": os.path.join(args.coco_dir, f), "label": "0",
              "generator": "", "source": "coco-val2017-orig"}
             for f in imgs[: args.cap]]
    out_rows = fakes + reals
    random.Random(args.seed + 1).shuffle(out_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(fakes)} fakes + {len(reals)} original-res reals -> {args.out}")


if __name__ == "__main__":
    main()

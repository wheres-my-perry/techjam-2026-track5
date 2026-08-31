"""Deliverable 5: contact sheets of representative false positives / false negatives.

`src/evaluate.py` dumps errors_clean.json as {"false_positives": [[path, score], ...],
"false_negatives": [...]}. This turns those into two labelled image grids so the error
analysis shows the actual failures rather than describing them.

Shows the ORIGINAL image where the manifest records one (`orig`), because a 176px
canonical crop tells a reader nothing about why a photo was mistaken for AI. The score
printed on each tile is the model's confidence that the image is AI-generated.

    python -m scripts.error_sheet --eval outputs/pe_ft/eval_canon6_test \
        --manifest data/manifests/canon6_test.csv --out error_analysis/canon6
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from PIL import Image, ImageDraw

TILE = 220
COLS = 5


def sheet(items, orig_of, title, out_png):
    if not items:
        print(f"  {title}: none")
        return
    items = items[:COLS * 3]
    rows = (len(items) + COLS - 1) // COLS
    W, H = COLS * TILE, rows * (TILE + 22) + 26
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), title, fill="black")
    for i, (p, score) in enumerate(items):
        src = orig_of.get(p, p)
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
        im.thumbnail((TILE - 8, TILE - 8), Image.LANCZOS)
        x = (i % COLS) * TILE + 4
        y = (i // COLS) * (TILE + 22) + 26
        canvas.paste(im, (x, y))
        d.text((x, y + TILE - 14), f"pred {score:.3f}", fill="black")
        d.text((x, y + TILE - 2), os.path.basename(src)[:28], fill="#555555")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png)
    print(f"  {title}: {len(items)} tiles -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="evaluation output dir")
    ap.add_argument("--manifest", default=None, help="manifest, to recover original files")
    ap.add_argument("--out", required=True, help="output dir")
    a = ap.parse_args()

    errs = json.load(open(os.path.join(a.eval, "errors_clean.json")))
    orig_of = {}
    if a.manifest and os.path.exists(a.manifest):
        for r in csv.DictReader(open(a.manifest, newline="")):
            if r.get("orig"):
                orig_of[r["path"]] = r["orig"]

    os.makedirs(a.out, exist_ok=True)
    sheet(errs.get("false_positives", []), orig_of,
          "FALSE POSITIVES - authentic photos the model called AI", os.path.join(a.out, "FP_real_called_AI.png"))
    sheet(errs.get("false_negatives", []), orig_of,
          "FALSE NEGATIVES - AI images the model called real", os.path.join(a.out, "FN_AI_called_real.png"))

    with open(os.path.join(a.out, "worst.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "path", "orig", "pred"])
        for kind, key in (("false_positive", "false_positives"), ("false_negative", "false_negatives")):
            for p, s in errs.get(key, []):
                w.writerow([kind, p, orig_of.get(p, ""), f"{s:.4f}"])
    print(f"  wrote {a.out}/worst.csv")


if __name__ == "__main__":
    main()

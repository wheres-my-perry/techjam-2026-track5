"""Canonicalize a dataset: seeded-random CROP at native resolution (+ optional
downscale-only pre-band for oversized images).

    python -m scripts.canonicalize --manifest data/manifests/wildfake_train.csv \
        --out-dir data/canon/wf_train --out-manifest data/manifests/canon_wf_train.csv \
        --crop 176

Kills the size->label shortcut (2026-08-28): every image, both classes, ends
up the SAME fixed size via a random-position crop of native pixels — no
resampling, so no resample signature to learn (Thinh's design). For datasets
with oversized images, --band MIN MAX first downscales (never upscales,
seeded random target, LANCZOS) into the range before cropping; downscale
traces only ever appear in eval-only sets. Per-path seeded rng: deterministic
per image, statistically independent of the label. Output PNG (one uniform
format). Skips existing files (resume-free).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import time

from PIL import Image

from src.data import load_image, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--crop", type=int, required=True,
                    help="final square crop size (native pixels)")
    ap.add_argument("--band", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="optional downscale-only pre-band for oversized images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap (seeded subsample) for quick runs")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for i, s in enumerate(samples):
        h = hashlib.md5(s.path.encode()).hexdigest()[:16]
        out_path = os.path.join(args.out_dir, f"{h}.png")
        if not os.path.exists(out_path):
            try:
                img = load_image(s.path)
            except Exception as e:
                print(f"skip {s.path}: {e}", flush=True)
                continue
            rng = random.Random(f"{args.seed}|{s.path}")
            w, hgt = img.size
            if args.band and min(w, hgt) > args.band[1]:
                target = rng.randint(args.band[0], args.band[1])
                sc = target / min(w, hgt)
                img = img.resize((max(1, round(w * sc)),
                                  max(1, round(hgt * sc))), Image.LANCZOS)
                w, hgt = img.size
            c = args.crop
            if min(w, hgt) < c:
                print(f"skip {s.path}: smaller than crop ({w}x{hgt})", flush=True)
                continue
            x = rng.randint(0, w - c)
            y = rng.randint(0, hgt - c)
            img = img.crop((x, y, x + c, y + c))
            tmp = out_path + ".tmp.png"
            img.save(tmp, format="PNG")
            os.replace(tmp, out_path)
        rows.append({"path": out_path, "label": s.label,
                     "generator": s.generator, "source": s.source})
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(samples) - i - 1) / max(rate, 1e-9)
            print(f"{i+1}/{len(samples)} ({rate:.0f}/s, eta {eta/60:.0f}m)",
                  flush=True)

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    with open(args.out_manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out_manifest}")


if __name__ == "__main__":
    main()

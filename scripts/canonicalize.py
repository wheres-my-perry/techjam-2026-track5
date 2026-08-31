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
import io
import os
import random
import time

from PIL import Image

from src.data import load_image, load_manifest


def _one(job):
    path, label, generator, source, out_dir, crop, band, long_side, jpeg_fakes, seed = job
    h = hashlib.md5(path.encode()).hexdigest()[:16]
    out_path = os.path.join(out_dir, f"{h}.png")
    native = 0
    if not os.path.exists(out_path):
        try:
            img = load_image(path)
        except Exception as e:
            print(f"skip {path}: {e}", flush=True)
            return None
        native = max(img.size)          # BEFORE any resize: the size-bucket key
        rng = random.Random(f"{seed}|{path}")
        if jpeg_fakes and label == 1:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=rng.randint(75, 95), subsampling=0)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        w, hgt = img.size
        if long_side and max(w, hgt) > long_side:
            sc = long_side / max(w, hgt)
            img = img.resize((max(1, round(w * sc)), max(1, round(hgt * sc))), Image.LANCZOS)
            w, hgt = img.size
        if band and min(w, hgt) > band[1]:
            target = rng.randint(band[0], band[1])
            sc = target / min(w, hgt)
            img = img.resize((max(1, round(w * sc)), max(1, round(hgt * sc))), Image.LANCZOS)
            w, hgt = img.size
        c = crop
        if min(w, hgt) < c:
            print(f"skip {path}: smaller than crop ({w}x{hgt})", flush=True)
            return None
        x = rng.randint(0, w - c)
        y = rng.randint(0, hgt - c)
        img = img.crop((x, y, x + c, y + c))
        tmp = out_path + f".{os.getpid()}.tmp.png"
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)
    if not native:
        # resume path: the canonical PNG already existed, so read the native long
        # side from the ORIGINAL header (lazy, no decode) -- the per-bucket balance
        # gate needs it for every row, not just freshly canonicalized ones.
        try:
            with Image.open(path) as im:
                native = max(im.size)
        except Exception as e:
            print(f"size-probe failed {path}: {e}", flush=True)
            return None
    return {"path": out_path, "orig": path, "label": label, "generator": generator,
            "source": source, "long": native}


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
    ap.add_argument("--long", type=int, default=0,
                    help="shrink so the LONG side == this (LANCZOS) when larger; "
                         "never upscale. Deterministic factor = long/LONG, so it is "
                         "label-neutral iff every native-size bucket holds both "
                         "classes (bucket_audit.py). Thinh's rule 2026-08-29.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-fakes", action="store_true",
                    help="give every FAKE one JPEG pass (q 75-95, seeded per path) "
                         "before cropping. Equalizes compression history: reals "
                         "arrive with >=1 generation (camera/web), diffusion fakes "
                         "with 0 (born PNG) -- found label-predictive 2026-08-29.")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap (seeded subsample) for quick runs")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    jobs = [(s.path, s.label, s.generator, s.source, args.out_dir, args.crop,
             args.band, args.long, args.jpeg_fakes, args.seed) for s in samples]
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(_one, jobs, chunksize=64)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(samples) - i - 1) / max(rate, 1e-9)
                print(f"{i+1}/{len(samples)} ({rate:.0f}/s, eta {eta/60:.0f}m)", flush=True)

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    with open(args.out_manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "orig", "label", "generator", "source", "long"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out_manifest}")


if __name__ == "__main__":
    main()

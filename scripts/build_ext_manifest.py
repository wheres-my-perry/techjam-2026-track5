"""Extract the large-image expansion sources (2026-08-29) into files + one raw manifest.

Why: every image in canon2 is 200-511 px native, so the model never saw large
content of EITHER class (0/10 on wild phone photos + Gemini images). Thinh's rule
for the fix: shrink everything to one small size first, then crop as before --
legal only if every native-size bucket holds both classes in equal amounts.
This script only extracts; balance is enforced in merge_ext.py and checked by
bucket_audit.py.

Labels: NEVER from folder names or a dataset's own 'label' column unless that
column is the real/fake label (SID_Set: 0 real / 1 full synthetic / 2 tampered).
CelebA-HQ / AFHQ 'label' columns are attribute classes -- ignored; those sets
are all-real by construction (documented sources).

Usage:  python -m scripts.build_ext_manifest [--out data/manifests/raw_ext.csv]
Idempotent: re-run as more shards land; existing files are kept.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
from multiprocessing import Pool

import pyarrow.parquet as pq
from PIL import Image

OUT_ROOT = "data/ext/img"

# name -> (glob, kind, label, generator, cap)
SOURCES = {
    "sid":             ("data/sid_set/data/*.parquet",                 "sid",      None, None,            None),
    "celebahq_1024":   ("data/ext/celeba-hq/data/*.parquet",           "plain",    0,    "",              9000),
    "afhq_512":        ("data/ext/AFHQv2/data/*.parquet",              "plain",    0,    "",              15000),
    "openimages_1024": ("data/ext/open-images-v7-subset/data/*.parquet","plain",   0,    "",              9000),
    "midjourney_v6":   ("data/ext/midjourney-v6-recap/train_*.parquet","plain",    1,    "midjourney_v6", 9000),
    "elsa":            ("data/ext/ELSA_D3/data/*.parquet",             "elsa",     None, None,            None),
    "ffhq_1024":       ("data/ext/ffhq-dataset/Part1/*.png",           "files",    0,    "",              None),
}
ELSA_GEN = {"DeepFloyd/IF-II-L-v1.0": "deepfloyd_if", "CompVis/stable-diffusion-v1-4": "sd14",
            "stabilityai/stable-diffusion-2-1-base": "sd21", "stabilityai/stable-diffusion-xl-base-1.0": "sdxl"}
SID = {0: (0, "", "sid_real"), 1: (1, "flux_sid", "sid_fake"), 2: (1, "sid_tampered", "sid_tampered")}


def write_bytes(b: bytes, out_dir: str, stem: str):
    """Write the ORIGINAL bytes (no re-encode: keeps native compression history)."""
    im = Image.open(io.BytesIO(b))
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(im.format, "png")
    p = os.path.join(out_dir, f"{stem}.{ext}")
    if not os.path.exists(p):
        with open(p + ".tmp", "wb") as fh:
            fh.write(b)
        os.replace(p + ".tmp", p)
    return p, im.size


def do_shard(job):
    name, shard, kind, label, generator = job
    rows = []
    stem0 = os.path.splitext(os.path.basename(shard))[0]
    if kind == "sid":
        t = pq.read_table(shard, columns=["img_id", "image", "label"]).to_pylist()
        for r in t:
            lab, gen, src = SID[int(r["label"])]
            out_dir = os.path.join(OUT_ROOT, src); os.makedirs(out_dir, exist_ok=True)
            p, (w, h) = write_bytes(r["image"]["bytes"], out_dir, r["img_id"])
            rows.append((p, lab, gen, src, w, h))
    elif kind == "elsa":
        cols = [f"image_gen{i}" for i in range(4)] + [f"model_gen{i}" for i in range(4)] + ["id"]
        t = pq.read_table(shard, columns=cols).to_pylist()
        for r in t:
            for i in range(4):
                gen = ELSA_GEN.get(r[f"model_gen{i}"], r[f"model_gen{i}"].split("/")[-1])
                src = "elsa_" + gen
                out_dir = os.path.join(OUT_ROOT, src); os.makedirs(out_dir, exist_ok=True)
                p, (w, h) = write_bytes(r[f"image_gen{i}"]["bytes"], out_dir, f"{r['id']}_g{i}")
                rows.append((p, 1, gen, src, w, h))
    else:  # plain: one 'image' column, class fixed by source
        out_dir = os.path.join(OUT_ROOT, name); os.makedirs(out_dir, exist_ok=True)
        t = pq.read_table(shard, columns=["image"]).to_pylist()
        for j, r in enumerate(t):
            p, (w, h) = write_bytes(r["image"]["bytes"], out_dir, f"{stem0}_{j:05d}")
            rows.append((p, label, generator, name, w, h))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests/raw_ext.csv")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    jobs, rows = [], []
    for name, (pat, kind, label, generator, cap) in SOURCES.items():
        fs = sorted(glob.glob(pat))
        if not fs:
            print(f"{name}: nothing landed yet"); continue
        if kind == "files":
            for f in fs:
                with Image.open(f) as im: w, h = im.size
                rows.append((f, label, generator, name, w, h))
            print(f"{name}: {len(fs)} files")
        else:
            jobs += [(name, f, kind, label, generator) for f in fs]
    with Pool(args.workers) as pool:
        for out in pool.imap_unordered(do_shard, jobs):
            rows += out
    # per-source caps (seeded by path order, deterministic)
    by = {}
    for r in rows: by.setdefault(r[3], []).append(r)
    final = []
    for src, rs in sorted(by.items()):
        cap = next((c for n, (_, _, _, _, c) in SOURCES.items() if n == src), None)
        rs = sorted(rs)
        if cap and len(rs) > cap: rs = rs[:cap]
        final += rs
        print(f"{src:18s} n={len(rs):6d}  label={rs[0][1]}  gen={rs[0][2] or '-':14s} sizes={sorted({(w, h) for *_, w, h in rs})[:4]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "generator", "source", "w", "h"]); w.writerows(final)
    print(f"{len(final)} rows -> {args.out}")


if __name__ == "__main__":
    main()

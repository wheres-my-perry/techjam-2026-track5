"""Build an unseen-GENERATOR test set for canon6 from bitmind mirrors.

Why this exists: canon4's headline ("0.9955 AUROC, 94% caught at 1% false alarms")
was measured on randtest_eq -- 11,729 unique images over 64 generator tags. That set
died with the server and cannot be rebuilt: docs/DATA_AUDIT names its sources only by
CATEGORY and extract_randtest.py is not in the repo. Several of those categories were
bitmind mirrors, which do still exist, so this rebuilds a comparable (NOT identical)
set from them. Any number from it is a canon6 number on a NEW set and must never be
presented as a reproduction of 0.9955.

Every generator here is absent from canon6 train/val. Reals are from sources canon6
never trained on. Byte- and perceptual-duplicate rows against training are dropped,
because the original set turned out to be 31% duplicates and that inflated its numbers.

    python -m scripts.build_unseen6 --fetch      # download shards
    python -m scripts.build_unseen6 --extract    # parquet -> files + raw manifest
"""
from __future__ import annotations

import argparse
import csv
import io
import os

ROOT = "/workspace/techjam-2026-track5"
OUT_ROOT = "data/unseen6/img"

# local name -> (repo id, n_shards, label, generator)
SOURCES = {
    "mobius":        ("bitmind/bm-mobius",                  2, 1, "mobius"),
    "realvisxl":     ("bitmind/bm-realvisxl",               2, 1, "realvis_xl"),
    "bmdiffusion":   ("bitmind/bm-diffusion",               2, 1, "bm_diffusion"),
    "ldm_face":      ("bitmind/DiffFace-LDM",               2, 1, "ldm_diffface"),
    "flux_celeba":   ("bitmind/celeb-a-hq___FLUX.1-dev",    2, 1, "flux1_dev"),
    "flux_coco":     ("bitmind/MS-COCO-unique___FLUX.1-dev", 2, 1, "flux1_dev"),
    "real_bm":       ("bitmind/bm-real",                    3, 0, ""),
    "real_diffface": ("bitmind/DiffFace-Real",              2, 0, ""),
}

# diffusers-parti-prompts: one repo per NAMED generator, the same prompts rendered by each.
# Ideal as an overfit checker (Thinh 2026-08-31): generators collected from the internet that
# canon6 never trained on. Split deliberately into two kinds, because they answer different
# questions and must not be pooled into one number:
#   UNSEEN ARCHITECTURE       - a family canon6 has never seen at all (the real generalisation test)
#   UNSEEN VERSION            - a different release of a family canon6 DOES train on (easier)
PARTI_ARCH = {                       # unseen architecture
    "parti_karlo":      "karlo-v1",
    "parti_kandinsky":  "kandinsky-2-2",
    "parti_wuerstchen": "wuerstchen",
    "parti_muse512":    "muse512",
    "parti_muse256":    "muse256",
    "parti_if":         "if-v-1.0",          # DeepFloyd-IF: routed test-only in canon6
}
PARTI_VERSION = {                    # unseen version of a family canon6 trains on
    "parti_sd15":          "sd-v1-5",
    "parti_sd21":          "sd-v2.1",
    "parti_sdxl09":        "sdxl-0.9",
    "parti_sdxl09_ref":    "sdxl-0.9-refiner",
    "parti_sdxl10":        "sdxl-1.0",
    "parti_sdxl10_ref":    "sdxl-1.0-refiner",
}
for _tag, _repo in {**PARTI_ARCH, **PARTI_VERSION}.items():
    SOURCES[_tag] = (f"diffusers-parti-prompts/{_repo}", 1, 1, _tag.replace("parti_", ""))


def fetch():
    from huggingface_hub import snapshot_download
    for name, (repo, n, _, _) in SOURCES.items():
        d = os.path.join(ROOT, "data/unseen6/raw", name)
        os.makedirs(d, exist_ok=True)
        pats = None if name.startswith("parti_") else \
            [f"data/train-{i:05d}-*.parquet" for i in range(n)]
        print(f"=== {repo} -> {name} ({n} shards)", flush=True)
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=d,
                              allow_patterns=pats, max_workers=8)
            print(f"=== DONE {repo}", flush=True)
        except Exception as e:
            print(f"=== FAIL {repo}: {type(e).__name__}: {e}", flush=True)


def _image_column(tbl):
    """Name of the binary-image column.

    "images" (plural) matters: every diffusers-parti-prompts repo uses it, and without it all
    twelve generators silently extracted zero images while the run still reported success.
    """
    for c in ("image", "images", "img", "jpg", "png", "image_bytes"):
        if c in tbl.column_names:
            return c
    return None


def extract(cap_per_source):
    import glob
    import pyarrow.parquet as pq
    from PIL import Image

    rows, missing_col = [], []
    for name, (repo, _, label, gen) in SOURCES.items():
        shards = sorted(glob.glob(os.path.join(ROOT, "data/unseen6/raw", name, "data", "*.parquet")))
        out_dir = os.path.join(ROOT, OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for shard in shards:
            if n >= cap_per_source:
                break
            t = pq.read_table(shard)
            col = _image_column(t)
            if col is None:
                print(f"  !! {name}: NO IMAGE COLUMN in {os.path.basename(shard)} "
                      f"(cols={t.column_names[:8]}) -- this source contributes NOTHING",
                      flush=True)
                missing_col.append(name)
                break
            for rec in t.column(col).to_pylist():
                if n >= cap_per_source:
                    break
                b = rec["bytes"] if isinstance(rec, dict) else rec
                if not b:
                    continue
                try:
                    im = Image.open(io.BytesIO(b))
                    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(im.format, "png")
                    w, h = im.size
                except Exception:
                    continue
                p = os.path.join(out_dir, f"{name}_{n:06d}.{ext}")
                if not os.path.exists(p):
                    with open(p, "wb") as fh:
                        fh.write(b)
                rows.append({"path": os.path.relpath(p, ROOT), "label": label,
                             "generator": gen, "source": f"unseen_{name}", "w": w, "h": h})
                n += 1
        print(f"  {name:14s} {n:6d} images  (label {label}, generator {gen or '-'})", flush=True)

    out = os.path.join(ROOT, "data/manifests/raw_unseen6.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source", "w", "h"])
        w.writeheader(); w.writerows(rows)
    nr = sum(1 for r in rows if r["label"] == 0)
    print(f"\n{len(rows)} rows ({nr} real / {len(rows)-nr} fake) -> {out}")
    got = {r["source"].replace("unseen_", "") for r in rows}
    empty = [n for n in SOURCES if n not in got]
    if empty:
        print(f"!! {len(empty)} source(s) produced NO images: {empty}")
    if missing_col:
        print(f"!! unrecognised image column in: {missing_col}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--cap", type=int, default=1500, help="images per source")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.extract:
        extract(a.cap)
    if not (a.fetch or a.extract):
        raise SystemExit("need --fetch and/or --extract")


if __name__ == "__main__":
    main()

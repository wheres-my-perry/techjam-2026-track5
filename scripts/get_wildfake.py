"""Pull WildFake subsets from ModelScope and build manifests (CSV-driven).

Layout learned 2026-08-26: images ship as per-generator ZIPs under Images/,
with authoritative per-generator label CSVs in label_csv_files/
(columns: Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num).
Official benchmark: dalle3.csv (8,843 fakes) + real_coco.csv val2017 rows (4,998 reals).

Typical use (machine with internet; pip install modelscope):

  python scripts/get_wildfake.py --list
  python scripts/get_wildfake.py --include 'Images/Real/coco.zip' \
      --extract-filter val2017 --delete-zips
  python scripts/get_wildfake.py --include 'Images/Diffusion_based/DDIM.zip' \
      --include 'Images/Real/imagenet.zip' --delete-zips
  python scripts/get_wildfake.py --manifest --holdout-generator ddim
  python scripts/get_wildfake.py --official-val     # needs dalle3 images + coco val2017

Disk notes: --extract-filter keeps only zip members whose path contains the given
substring(s); --delete-zips removes each zip after successful extraction.
Hard rule: dalle3 + coco val2017 never enter train/val/test manifests
(they are the official benchmark -> official_val.csv only).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ID = "hy2628982280/WildFake"
LOCAL_DIR = os.path.join(ROOT, "data", "wildfake", "raw")
LABEL_DIR = os.path.join(LOCAL_DIR, "label_csv_files")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORBIDDEN_CSVS = {"dalle3.csv"}          # entire csv is official benchmark
VAL2017_MARKER = "val2017"               # forbidden rows inside real_coco.csv


# ------------------------------------------------------------------ listing

def list_files():
    def _name(f):
        if isinstance(f, str):
            return f
        if isinstance(f, dict):
            return f.get("Path") or f.get("Name") or str(f)
        for attr in ("path", "file_path", "rfilename", "name"):
            v = getattr(f, attr, None)
            if v:
                return v
        return str(f)

    def _size(f):
        for attr in ("size", "Size", "file_size"):
            v = getattr(f, attr, None) if not isinstance(f, dict) else f.get(attr)
            if isinstance(v, (int, float)):
                return int(v)
        return -1

    def human(b):
        if b < 0:
            return "?"
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.1f}{u}"
            b /= 1024
        return f"{b:.1f}PB"

    from modelscope.hub.api import HubApi
    files = HubApi().list_repo_files(DATASET_ID, repo_type="dataset")
    print(f"{len(files)} files (all, with sizes):")
    for f in sorted(files, key=_name):
        print(f"  {human(_size(f)):>9s}  {_name(f)}")


# ----------------------------------------------------------------- download

def download(includes, extract_filters=(), delete_zips=False):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    cmd = ["modelscope", "download", "--repo-type", "dataset", DATASET_ID,
           "--local-dir", LOCAL_DIR, "--include", *includes]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    filters = [f.lower() for f in extract_filters]
    for dirpath, _, files in os.walk(LOCAL_DIR):
        for fn in files:
            if not fn.lower().endswith(".zip"):
                continue
            p = os.path.join(dirpath, fn)
            out = os.path.splitext(p)[0]
            print(f"unzipping {p} -> {out}" + (f" (filter: {filters})" if filters else ""))
            with zipfile.ZipFile(p) as z:
                members = z.namelist()
                if filters:
                    members = [m for m in members
                               if any(s in m.lower() for s in filters)]
                    print(f"  {len(members)} of {len(z.namelist())} members match")
                z.extractall(out, members=members)
            if delete_zips:
                os.remove(p)
                print(f"  deleted {p}")


# ------------------------------------------------- disk index + csv reading

def norm_key(p):
    """Normalize any path into the WildFake CSV key space (no './', no 'Images/')."""
    p = p.replace(os.sep, "/")
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("Images/"):
        p = p[len("Images/"):]
    return p


def disk_index():
    """suffix -> [repo-relative paths] for every image file under LOCAL_DIR.

    Keyed by FULL PATH SUFFIXES, never by basename (fixed 2026-08-31). Every
    real_*.csv names its files img000000.jpg, so church/imagenet/ffhq/afhq/
    celebahq collide on basename; the old basename index silently kept one
    arbitrary winner per name, which is how 24.5% of claimed training fakes
    turned out to be real photos (docs/LESSONS_FOR_TEAMMATES.md S1).

    Every suffix of >=2 components is registered, so an extra directory level
    introduced by zip extraction still resolves, and any csv path that matches
    more than one file on disk is reported as AMBIGUOUS and dropped rather than
    guessed at.
    """
    idx = {}
    for dirpath, _, files in os.walk(LOCAL_DIR):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            parts = norm_key(os.path.relpath(full, LOCAL_DIR)).split("/")
            for i in range(len(parts) - 1, -1, -1):
                idx.setdefault("/".join(parts[i:]), []).append(rel)
    return idx


def lookup(idx, image_path):
    """Resolve one csv Image_path to a unique file on disk, or None."""
    hits = idx.get(norm_key(image_path))
    if not hits:
        return None
    if len(hits) > 1:
        return AMBIGUOUS
    return hits[0]


AMBIGUOUS = object()


def read_label_csv(path):
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


# ---------------------------------------------------------------- manifests

def build_manifests(val_frac=0.1, test_frac=0.1, seed=0, cap_per_group=20000,
                    holdout_generators=()):
    if not os.path.isdir(LABEL_DIR):
        sys.exit(f"{LABEL_DIR} missing — download label_csv_files/** first.")
    idx = disk_index()
    print(f"disk index: {len(idx)} images under {LOCAL_DIR}")

    rows, missing, ambiguous = [], 0, 0
    for csv_name in sorted(os.listdir(LABEL_DIR)):
        if not csv_name.endswith(".csv"):
            continue
        gen = os.path.splitext(csv_name)[0].lower()
        if csv_name in FORBIDDEN_CSVS:
            continue
        found_here = 0
        for r in read_label_csv(os.path.join(LABEL_DIR, csv_name)):
            ip = r.get("Image_path", "")
            if gen == "real_coco" and VAL2017_MARKER in ip.lower():
                continue  # official benchmark reals
            local = lookup(idx, ip)
            if local is None:
                missing += 1
                continue
            # BUG FIX 2026-08-30: filenames are NOT unique across WildFake (GAN images and the
            # real AFHQ/FFHQ photos are both img000000.jpg...). Matching by basename alone turned
            # every not-downloaded GAN row into a real photo labelled fake (24% of canon2..4
            # "fakes"). The 08-30 fix required the CSV's top-level folder to appear in the local
            # path; that stops fake-vs-real cross-matching but NOT collisions inside Real/, where
            # church/imagenet/ffhq/afhq/celebahq all ship img000000.jpg and the basename index
            # silently kept one arbitrary winner per name. Superseded 2026-08-31: lookup() now
            # resolves the FULL csv path against a suffix index, so a row either resolves to
            # exactly one file or is reported AMBIGUOUS and dropped.
            if local is AMBIGUOUS:
                ambiguous += 1
                continue
            label = int(r.get("IsFake", "1"))
            rows.append({"path": local, "label": label,
                         "generator": gen if label == 1 else "",
                         "source": "wildfake"})
            found_here += 1
        if found_here:
            print(f"  {csv_name}: {found_here} images on disk")
    if not rows:
        sys.exit("No labeled images found on disk — download some Images/ zips first.")
    print(f"{len(rows)} usable rows ({missing} csv rows not on disk — fine if "
          "you only downloaded some zips)")
    if ambiguous:
        print(f"WARNING: {ambiguous} csv rows matched >1 file on disk and were "
              "DROPPED (ambiguous label). Investigate before training.")

    rng = random.Random(seed)
    by_group: dict[tuple, list] = {}
    for r in rows:
        by_group.setdefault((r["label"], r["generator"]), []).append(r)
    kept = []
    for g, items in sorted(by_group.items()):
        rng.shuffle(items)
        kept.extend(items[:cap_per_group])
        print(f"  group {g}: {min(len(items), cap_per_group)} kept / {len(items)}")
    rng.shuffle(kept)

    ho = {h.lower() for h in holdout_generators}
    held = [r for r in kept if r["label"] == 1 and r["generator"] in ho]
    rest = [r for r in kept if not (r["label"] == 1 and r["generator"] in ho)]
    if ho:
        found = {r["generator"] for r in held}
        if ho - found:
            print(f"WARNING: holdout generator(s) not found: {sorted(ho - found)}")
        print(f"held-out {sorted(found)}: {len(held)} fakes -> test only")

    n = len(rest)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    splits = {"test": rest[:n_test] + held,
              "val": rest[n_test:n_test + n_val],
              "train": rest[n_test + n_val:]}
    rng.shuffle(splits["test"])
    man_dir = os.path.join(ROOT, "data", "manifests")
    os.makedirs(man_dir, exist_ok=True)
    for split, items in splits.items():
        out = os.path.join(man_dir, f"wildfake_{split}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source"])
            w.writeheader()
            w.writerows(items)
        n_fake = sum(r["label"] for r in items)
        gens = sorted({r["generator"] for r in items if r["label"] == 1})
        print(f"{out}: {len(items)} rows ({n_fake} fake / {len(items)-n_fake} real), "
              f"generators: {gens}")


def build_official_val_manifest():
    idx = disk_index()
    rows, miss_fake, miss_real = [], 0, 0
    p3 = os.path.join(LABEL_DIR, "dalle3.csv")
    pc = os.path.join(LABEL_DIR, "real_coco.csv")
    for r in read_label_csv(p3):
        local = lookup(idx, r.get("Image_path", ""))
        if local is None or local is AMBIGUOUS:
            miss_fake += 1
            continue
        rows.append({"path": local, "label": 1, "generator": "dalle_advanced",
                     "source": "official_val"})
    for r in read_label_csv(pc):
        ip = r.get("Image_path", "")
        if VAL2017_MARKER not in ip.lower():
            continue
        local = lookup(idx, ip)
        if local is None or local is AMBIGUOUS:
            miss_real += 1
            continue
        rows.append({"path": local, "label": 0, "generator": "",
                     "source": "official_val"})
    random.Random(0).shuffle(rows)  # never write class-sorted manifests
    n_fake = sum(r["label"] for r in rows)
    n_real = len(rows) - n_fake
    print(f"official val: {n_fake}/8843 fakes and {n_real}/4998 reals on disk "
          f"(missing: {miss_fake} fake, {miss_real} real)")
    if not rows:
        sys.exit("Nothing on disk — need DALLE.zip's dalle3 subtree and/or "
                 "coco.zip's val2017 subtree (see module docstring).")
    out = os.path.join(ROOT, "data", "manifests", "official_val.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--extract-filter", action="append", default=[],
                    help="only extract zip members whose path contains this substring")
    ap.add_argument("--delete-zips", action="store_true",
                    help="delete each zip after successful extraction (saves disk)")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--official-val", action="store_true")
    ap.add_argument("--holdout-generator", action="append", default=[])
    ap.add_argument("--cap-per-group", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list:
        list_files()
    if args.include:
        download(args.include, args.extract_filter, args.delete_zips)
    if args.manifest:
        build_manifests(seed=args.seed, cap_per_group=args.cap_per_group,
                        holdout_generators=args.holdout_generator)
    if args.official_val:
        build_official_val_manifest()
    if not (args.list or args.include or args.manifest or args.official_val):
        print(__doc__)


if __name__ == "__main__":
    main()

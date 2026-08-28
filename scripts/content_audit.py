"""Structural content audit: does every SUBJECT appear on both sides of the label?

    python -m scripts.content_audit --manifests data/manifests/canon2_train.csv \
        data/manifests/canon2_test.csv

The model-free companion to canary_audit. Canaries measure whether reals and
fakes differ in dumb pixel statistics; this script explains WHY, by tagging
every row with a coarse subject (faces, bedroom, church, animals, ...) from
its original path / ArtiFact category and printing real-vs-fake counts per
subject per split. A subject with hundreds of rows on one side and ~none on
the other is a content shortcut ("bedroom = fake") and is flagged.

Needs the `orig` column that scripts/canonicalize.py writes (2026-08-29+).
"""

from __future__ import annotations

import argparse
import csv
import os
import re

# keyword -> coarse subject; first match wins, checked against the lowercased
# "<source>/<category>/<orig path>" string. Order matters (more specific first).
RULES = [
    (r"bedroom|lsun-bed|/bed/", "bedroom"),
    (r"church|churces", "church"),
    (r"landscape", "landscape"),
    (r"monet|art_painting|painting", "painting"),
    (r"ffhq|celeb|metfaces|sfhq|face|star_gan|stargan|blond|black_hair|brown_hair|"
     r"/male|/female|/young", "faces"),
    (r"afhq|/cat/|/dog/|/wild/|imgs_cat|imgs_dog|imgs_wild|horse2zebra|/horse/|/cat$|/dog$|/wild$", "animals"),
    (r"/car/|cityscapes|car-part", "vehicles/street"),
    (r"coco|cc9k|imagenet|places|ade|/images/|/n0\d+|glide|latentdiff|stable_diffusion|t2i|tt-cc",
     "general scenes"),
    (r"pro_gan|/pro/", "voc objects"),
]


def load_artifact_categories(root="data/artifact/ArtiFact") -> dict[str, str]:
    cats = {}
    for parent in ("Real", "Fake"):
        pdir = os.path.join(root, parent)
        if not os.path.isdir(pdir):
            continue
        for src in os.listdir(pdir):
            md = os.path.join(pdir, src, "metadata.csv")
            if not os.path.isfile(md):
                continue
            with open(md, newline="") as fh:
                for r in csv.DictReader(fh):
                    cats[os.path.join(pdir, src, r["image_path"])] = \
                        f"{src}/{r.get('category', '') or ''}"
    return cats


def subject(row, cats) -> str:
    orig = row.get("orig", "") or row.get("path", "")
    key = f"{row.get('source', '')}/{cats.get(orig, '')}/{orig}".lower()
    for pat, name in RULES:
        if re.search(pat, key):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--min-rows", type=int, default=200,
                    help="flag a subject when one side has >= this and the "
                         "other side has < 10%% of it")
    args = ap.parse_args()
    cats = load_artifact_categories()
    any_flag = False
    for m in args.manifests:
        with open(m, newline="") as fh:
            rows = list(csv.DictReader(fh))
        table: dict[str, list[int]] = {}
        for r in rows:
            table.setdefault(subject(r, cats), [0, 0])[int(r["label"])] += 1
        print(f"\n{m}  ({len(rows)} rows)")
        print(f"  {'subject':18s} {'real':>8s} {'fake':>8s}  {'fake:real':>9s}  flag")
        for subj, (nr, nf) in sorted(table.items(), key=lambda kv: -(sum(kv[1]))):
            big, small = max(nr, nf), min(nr, nf)
            flag = ""
            if big >= args.min_rows and small < 0.1 * big:
                flag = f"ONE-SIDED -> '{subj} = {'fake' if nf > nr else 'real'}'"
                any_flag = True
            ratio = f"{nf / nr:.2f}" if nr else "inf"
            print(f"  {subj:18s} {nr:8d} {nf:8d}  {ratio:>9s}  {flag}")
    print("\nVERDICT:", "ONE-SIDED SUBJECTS FOUND — fix the data" if any_flag
          else "every subject has both classes")
    raise SystemExit(1 if any_flag else 0)


if __name__ == "__main__":
    main()

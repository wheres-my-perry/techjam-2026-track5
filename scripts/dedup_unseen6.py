"""Drop unseen6 rows that also appear in canon6 training data.

Mandatory before quoting any unseen-generator number. The original unseen-64 set
turned out to be 31% duplicate rows, and re-reading it on unique images moved the
headline materially -- so duplicates are not a rounding error here.

Two risks this closes:
  * REAL overlap: bitmind's bm-real / DiffFace-Real are built from the same public
    photo pools we train on (FFHQ, CelebA-HQ, COCO, Open Images). A real image in
    both sets would be scored as "never trained on" when it was.
  * FAKE overlap: mirrors of the same generated corpora.

Matching is on the ORIGINAL files, not the canonical crops: canonicalize takes a
per-path seeded random crop, so the same source image reached by two paths yields
two different crops and would not match after canonicalization.

    python -m scripts.dedup_unseen6 --raw data/manifests/raw_unseen6.csv \
        --train data/manifests/canon6_train.csv data/manifests/canon6_val.csv \
        --out data/manifests/raw_unseen6_unique.csv [--maxd 2]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def _hashes(path):
    try:
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        with Image.open(path) as im:
            g = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        d = int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)
        return path, sha, d
    except Exception:
        return path, None, None


def run(paths, workers):
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, sha, d in ex.map(_hashes, paths, chunksize=64):
            if sha is not None:
                out[p] = (sha, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxd", type=int, default=2, help="max dHash Hamming distance = duplicate")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.raw, newline="")))
    train_paths = []
    for m in a.train:
        for r in csv.DictReader(open(m, newline="")):
            train_paths.append(r.get("orig") or r["path"])
    train_paths = sorted(set(train_paths))
    print(f"hashing {len(train_paths)} training originals and {len(rows)} unseen images "
          f"({a.workers} workers)...", flush=True)

    th = run(train_paths, a.workers)
    uh = run([r["path"] for r in rows], a.workers)
    print(f"  hashed {len(th)} train / {len(uh)} unseen", flush=True)

    tsha = {v[0] for v in th.values()}
    # bucket training dHashes by their top 32 bits so near-dup search stays cheap
    buckets = {}
    for sha, d in th.values():
        buckets.setdefault(d >> 32, []).append(d)

    kept, drop_reason = [], Counter()
    for r in rows:
        h = uh.get(r["path"])
        if h is None:
            drop_reason["unreadable"] += 1
            continue
        sha, d = h
        if sha in tsha:
            drop_reason[f"byte-identical to train ({r['source']})"] += 1
            continue
        hit = False
        for shift in (d >> 32, (d >> 32) ^ 1):
            for td in buckets.get(shift, ()):
                if bin(d ^ td).count("1") <= a.maxd:
                    hit = True
                    break
            if hit:
                break
        if hit:
            drop_reason[f"near-duplicate of train ({r['source']})"] += 1
            continue
        kept.append(r)

    print("\nDROPPED:")
    for k, v in drop_reason.most_common():
        print(f"  {v:6d}  {k}")
    if not drop_reason:
        print("  (none)")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source", "w", "h"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(kept)
    per = Counter(r["source"] for r in kept)
    nr = sum(1 for r in kept if r["label"] == "0")
    print(f"\nkept {len(kept)} of {len(rows)} ({nr} real / {len(kept)-nr} fake) -> {a.out}")
    for k, v in sorted(per.items()):
        print(f"  {k:22s} {v:6d}")


if __name__ == "__main__":
    main()

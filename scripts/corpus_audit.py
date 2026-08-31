"""Corpus self-audit: the checks the separability gates do NOT do.

Written 2026-08-31 after canon6 was audited with five gates and still had a
one-sided subject ('bedroom = fake', 92.7:1) plus unchecked duplicates. The
existing gates answer "can a dumb model separate the classes?". They never ask
"is the corpus itself sound?". Modelled on the canon5 audit
(docs/DATA_AUDIT_2026-08-30.md sections F/G) and on the phases a general image
dataset-curation checklist runs (quality -> duplicates -> distribution -> leakage).

  1. DEGENERATE IMAGES  canon5 canonicalisation produced 78 blank 170-byte PNGs
     from corrupt originals, and they appeared under BOTH labels. Flat images
     also make dHash useless (all-zero), so they must be found before dedup.
  2. BYTE DUPLICATES    inside each split and, worse, ACROSS splits.
  3. LABEL CONFLICTS    the same bytes carrying both labels.
  4. NEAR-DUPLICATES    val/test rows that are perceptual copies of a train row
     (dHash Hamming <= 2). canon5 dropped 65 val + 321 test on this.
  5. READABILITY        files that fail to open at all.

    python -m scripts.corpus_audit --prefix data/manifests/canon6 [--maxd 2]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
SPLITS = ("train", "val", "test")


def probe(path):
    """(sha256, dhash, flat?, size_bytes, unreadable?)"""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        with Image.open(path) as im:
            g = ImageOps.exif_transpose(im).convert("L")
            a8 = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
            flat = float(np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float32).std()) < 1.0
        d = int("".join("1" if b else "0" for b in (a8[:, 1:] > a8[:, :-1]).flatten()), 2)
        return path, sha, d, flat, size, False
    except Exception:
        return path, None, None, False, 0, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--maxd", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--write-drop", default=None,
                    help="write every offending canonical path here, for build_canon6 --exclude")
    a = ap.parse_args()

    rows = {}
    for sp in SPLITS:
        p = f"{a.prefix}_{sp}.csv"
        if os.path.exists(p):
            rr = list(csv.DictReader(open(p, newline="")))
            rows[sp] = rr[:a.limit] if a.limit else rr
            print(f"{sp}: {len(rows[sp])} rows")
    allrows = [(sp, r) for sp in rows for r in rows[sp]]
    paths = [r["path"] for _, r in allrows]

    print(f"\nprobing {len(paths)} canonical files ({a.workers} workers)...", flush=True)
    info = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, sha, d, flat, size, bad in ex.map(probe, paths, chunksize=256):
            info[path] = (sha, d, flat, size, bad)

    fails = 0

    # 1 + 5 -------------------------------------------------------------
    unread = [p for p, v in info.items() if v[4]]
    flat = [p for p, v in info.items() if v[2] and not v[4]]
    print(f"\n1/5. DEGENERATE & UNREADABLE")
    print(f"  unreadable files : {len(unread)}")
    print(f"  flat/blank images: {len(flat)}")
    if flat:
        by = Counter()
        for sp, r in allrows:
            if r["path"] in set(flat):
                by[(sp, r["label"])] += 1
        print(f"    by split/label: {dict(by)}")
        print(f"    e.g. {flat[:3]}")
    if unread or flat:
        fails += 1
        print("    -> DROP these rows before training (canon5 had 78 such PNGs under BOTH labels)")

    # 2 + 3 -------------------------------------------------------------
    by_sha = defaultdict(list)
    for sp, r in allrows:
        sha = info[r["path"]][0]
        if sha:
            by_sha[sha].append((sp, r["path"], r["label"]))
    within = Counter()
    cross, conflict = 0, 0
    for sha, group in by_sha.items():
        if len(group) < 2:
            continue
        splits = {g[0] for g in group}
        labels = {g[2] for g in group}
        if len(labels) > 1:
            conflict += 1
        if len(splits) > 1:
            cross += 1
        for sp in splits:
            n = sum(1 for g in group if g[0] == sp)
            if n > 1:
                within[sp] += n - 1
    print(f"\n2/3. BYTE DUPLICATES")
    print(f"  duplicate copies within a split: {dict(within) or 'none'}")
    print(f"  files appearing in >1 split    : {cross}")
    print(f"  byte-identical LABEL CONFLICTS : {conflict}")
    if cross or conflict:
        fails += 1
        print("    -> cross-split duplicates leak train into val/test; label conflicts are unlabelable")

    # 4 -----------------------------------------------------------------
    print(f"\n4. PERCEPTUAL NEAR-DUPLICATES (dHash Hamming <= {a.maxd}, flat images excluded)")
    tr = [info[r["path"]][1] for sp, r in allrows
          if sp == "train" and info[r["path"]][1] is not None and not info[r["path"]][2]]
    buckets = defaultdict(list)
    for d in tr:
        buckets[d >> 40].append(d)
    for sp in ("val", "test"):
        if sp not in rows:
            continue
        hits = 0
        for r in rows[sp]:
            d, isflat = info[r["path"]][1], info[r["path"]][2]
            if d is None or isflat:
                continue
            for k in (d >> 40, (d >> 40) ^ 1):
                if any(bin(d ^ t).count("1") <= a.maxd for t in buckets.get(k, ())):
                    hits += 1
                    break
        pct = 100.0 * hits / max(1, len(rows[sp]))
        print(f"  {sp}: {hits} of {len(rows[sp])} rows ({pct:.2f}%) are near-copies of a TRAIN image")
        if pct > 0.5:
            fails += 1
            print("    -> drop them; they inflate val/test (canon5 dropped 65 val + 321 test)")

    if a.write_drop:
        drop = set(unread) | set(flat)
        for sha, group in by_sha.items():           # cross-split byte duplicates
            if len({g[0] for g in group}) > 1:
                drop.update(g[1] for g in group)
        for sp in ("val", "test"):                  # near-copies of a training image
            for r in rows.get(sp, []):
                d, isflat = info[r["path"]][1], info[r["path"]][2]
                if d is None or isflat:
                    continue
                for k in (d >> 40, (d >> 40) ^ 1):
                    if any(bin(d ^ t).count("1") <= a.maxd for t in buckets.get(k, ())):
                        drop.add(r["path"]); break
        with open(a.write_drop, "w") as fh:
            fh.write("\n".join(sorted(drop)) + "\n")
        print(f"\n  wrote {len(drop)} paths to drop -> {a.write_drop}")

    print(f"\nVERDICT: {'CLEAN' if not fails else str(fails) + ' PROBLEM AREA(S) — fix before training'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()

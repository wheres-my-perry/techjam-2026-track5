"""Assemble canon6 -- the canon4/canon5 recipe rebuilt from scratch (2026-08-31).

canon5 and every checkpoint lived only on the server that died; this rebuilds an
equivalent corpus from the sources re-fetched by get_wildfake.py / get_ext.py /
extract_artifact_subset.py. Same protocol rules as canon5, enforced here rather
than assumed:

  * HOLD-OUT BY GENERATOR NAME, ACROSS ALL SOURCES. ddpm ships in BOTH WildFake
    and ArtiFact; canon2 leaked it into train through ArtiFact's folder while it
    was "held out" of WildFake. Test-only routing keys on the generator name, so
    a generator cannot re-enter train through a second dataset.
  * PARTIAL EDITS ARE TEST-ONLY. lama / mat / generative_inpainting / palette /
    sid_tampered are localized edits; a whole-image label is wrong for them.
  * SPLIT BY SOURCE FILE, never by row, so one file cannot span two splits.
  * PER-BUCKET CLASS BALANCE in train and val, bucketed by NATIVE long side. The
    shrink-to-320 step is only legal if "was shrunk by factor f" is independent
    of the label, which requires real == fake inside every native-size bucket.
    Excess goes to test rather than being discarded.
  * --cap-bucket keeps the <=341 bucket (ArtiFact/WildFake thumbnails, which
    otherwise dominate) from crowding out the large-image buckets that gave
    canon3+ its real-photo competence.

    python -m scripts.build_canon6 --canon data/manifests/canon_artifact.csv \
        data/manifests/canon_ext.csv --out-prefix data/manifests/canon6
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict

FIELDS = ["path", "orig", "label", "generator", "source", "long"]

# Held out from train/val entirely. Keyed on the generator name, across sources.
HOLDOUT = {"ddpm"}
PARTIAL_EDIT = {"sid_tampered", "lama", "mat", "generative_inpainting", "palette"}
TEST_ONLY_GEN = HOLDOUT | PARTIAL_EDIT | {"deepfloyd_if"}


def bucket(long_side: int) -> str:
    if long_side <= 341: return "<=341"
    if long_side <= 512: return "342-512"
    if long_side <= 768: return "513-768"
    if long_side <= 1024: return "769-1024"
    return ">1024"


def balance(rows, seed, tag, cap=0):
    """Per bucket keep min(n_real, n_fake) (at most `cap`) of each class."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["bucket"]][r["label"]].append(r)
    kept, excess, rng = [], [], random.Random(seed)
    for b, cls in sorted(by.items()):
        real, fake = cls.get("0", []), cls.get("1", [])
        n = min(len(real), len(fake))
        if cap:
            n = min(n, cap)
        for rs in (real, fake):
            rs = rs[:]
            rng.shuffle(rs)
            kept += rs[:n]
            excess += rs[n:]
        print(f"  {tag:5s} bucket {b:9s} real {len(real):7d} fake {len(fake):7d} "
              f"-> keep {n:6d} each ({len(real) + len(fake) - 2 * n} to test)")
    return kept, excess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", nargs="+", required=True,
                    help="canonicalized manifests (path,orig,label,generator,source,long)")
    ap.add_argument("--out-prefix", default="data/manifests/canon6")
    ap.add_argument("--cap-bucket", type=int, default=45000,
                    help="max rows per class per bucket in train (0 = uncapped)")
    ap.add_argument("--exclude", default=None,
                    help="file of canonical paths to drop (from corpus_audit --write-drop): "
                         "blank/flat images, cross-split byte duplicates, and val/test rows that "
                         "are perceptual copies of a training image")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    excluded = set()
    if a.exclude and os.path.exists(a.exclude):
        excluded = {l.strip() for l in open(a.exclude) if l.strip()}
        print(f"exclusion list: {len(excluded)} canonical paths from {a.exclude}")

    rows, seen, n_excluded = [], set(), 0
    for p in a.canon:
        n_before = len(rows)
        for r in csv.DictReader(open(p, newline="")):
            if not r.get("long") or int(r["long"]) <= 0:
                continue
            if r["path"] in excluded:      # audited-bad rows (blank, dup, near-dup)
                n_excluded += 1
                continue
            if r["orig"] in seen:          # same source file via two manifests
                continue
            seen.add(r["orig"])
            r["bucket"] = bucket(int(r["long"]))
            rows.append(r)
        print(f"{p}: +{len(rows) - n_before} rows")
    print(f"total {len(rows)} rows from {len(a.canon)} manifests"
          + (f" ({n_excluded} dropped by --exclude)" if n_excluded else ""))

    # ---- route + split by (source, generator), splitting whole source files ----
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["generator"])].append(r)

    tr, va, te = [], [], []
    for i, (key, rs) in enumerate(sorted(groups.items())):
        gen = key[1]
        if gen in TEST_ONLY_GEN:
            te += rs
            why = "HOLD-OUT" if gen in HOLDOUT else (
                  "partial-edit" if gen in PARTIAL_EDIT else "test-only")
            print(f"  {str(key):46s} n={len(rs):7d} -> TEST ONLY ({why})")
            continue
        rs = rs[:]
        random.Random(a.seed + i).shuffle(rs)
        n = len(rs)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        tr += rs[:n_tr]; va += rs[n_tr:n_tr + n_va]; te += rs[n_tr + n_va:]

    print(f"pre-balance: train {len(tr)} val {len(va)} test {len(te)}")
    print("balance (per native-size bucket, larger class subsampled):")
    tr, x1 = balance(tr, a.seed + 100, "train", a.cap_bucket)
    va, x2 = balance(va, a.seed + 101, "val", max(1, a.cap_bucket // 8) if a.cap_bucket else 0)
    te += x1 + x2

    for split, rs in (("train", tr), ("val", va), ("test", te)):
        random.Random(a.seed).shuffle(rs)
        out = f"{a.out_prefix}_{split}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(rs)
        nr = sum(r["label"] == "0" for r in rs)
        gens = len({r["generator"] for r in rs if r["label"] == "1"})
        print(f"{out}: {len(rs)} rows ({nr} real / {len(rs) - nr} fake), {gens} fake generators")


if __name__ == "__main__":
    main()

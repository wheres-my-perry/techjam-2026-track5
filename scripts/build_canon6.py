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
#
# ddim added 2026-08-31 (Thinh's rule: one-sided content may be kept for TESTING but never for
# TRAINING). Measured on canon6_train, ddim is 19,093 rows -- 30% of the fake class -- and its
# content is bedroom 76.4% / church 23.6%, i.e. 100% in two subjects. Two separate harms:
#   * shortcut: 'bedroom => fake' (content_audit measured 92.7:1 before LSUN reals were added);
#   * competence: a model whose fake class is one-third bedrooms learns to detect BEDROOMS, and
#     meets a phone photo of a person with nothing. That is why canon2 scored 0/10 on wild images.
# Balancing it was not enough: adding real bedrooms only moves the skew to 2.14:1, and removing
# ddim while KEEPING those reals would flip the axis to 'bedroom => real'. Both sides of the
# bedroom/church axis therefore leave training together (see TEST_ONLY_SOURCE).
HOLDOUT = {"ddpm", "ddim"}
PARTIAL_EDIT = {"sid_tampered", "lama", "mat", "generative_inpainting", "palette"}
TEST_ONLY_GEN = HOLDOUT | PARTIAL_EDIT | {"deepfloyd_if"}

# Real sources routed test-only.
TEST_ONLY_SOURCE = set()

# Per-source caps applied BEFORE the size-bucket balance, for sources that exist to balance a
# SUBJECT rather than a size bucket. Holding ddim/ddpm out does not remove bedrooms from the fake
# class: ArtiFact's LSUN-trained GANs (diffusion_gan, denoising_diffusion_gan, stable_diffusion)
# still emit ~1,400. Measured both extremes -- all 20K LSUN bedroom reals in train gives
# 'bedroom = fake' 2.14:1 while ddim is present, and none gives 12.55:1 once it is removed. The
# cap admits just enough real bedrooms to match the residual fake ones, so the subject is
# two-sided without flooding the <=341 bucket and crowding out other content.
CAP_SOURCE = {
    "lsun_bedroom": 3000,
    # afhq_512 is 15,000 animal close-ups at native 512, i.e. the ENTIRE 342-512 real side.
    # Sampling that bucket showed 10 of 12 reals were cat/dog faces while the fakes were diverse
    # commercial scenes -- size-balanced (bucket_audit 1.00) and content-disjoint. Capped so the
    # diverse web photos (flickr30k_web, native ~500x375) carry that bucket instead.
    "afhq_512": 4000,
    # celebahq_1024 was 4,008 of the 8,943 reals in the 769-1024 bucket and ALL faces, while that
    # bucket's fakes (midjourney_v6, flux_sid) contain almost none -- "1024px face => real".
    # Capped so openimages_1024 (diverse web photography) carries the bucket instead.
    "celebahq_1024": 1500,
}


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

    for src, cap in sorted(CAP_SOURCE.items()):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        if len(idx) > cap:
            rng = random.Random(a.seed + 7)
            drop = set(rng.sample(idx, len(idx) - cap))
            rows = [r for i, r in enumerate(rows) if i not in drop]
            print(f"  source cap: {src} {len(idx)} -> {cap} (subject-balance cap, see CAP_SOURCE)")

    # ---- route + split by (source, generator), splitting whole source files ----
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["generator"])].append(r)

    tr, va, te = [], [], []
    for i, (key, rs) in enumerate(sorted(groups.items())):
        src, gen = key
        if src in TEST_ONLY_SOURCE:
            te += rs
            print(f"  {str(key):46s} n={len(rs):7d} -> TEST ONLY (source paired with a held-out generator)")
            continue
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

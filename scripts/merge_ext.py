"""canon3 = canon2 + the large-image expansion (canon_ext), with Thinh's
per-bucket balance rule enforced (2026-08-29):

  shrink everything to one size first (canonicalize --long), then crop as
  before -- legal only if every native-size bucket holds both classes in equal
  amounts, otherwise "was shrunk by factor f" becomes a label.

Buckets are by native LONG side (that is what --long scales by). canon2 rows
are all <=341 px long side (verified 2026-08-29: no orig >= 512 short side,
ArtiFact/WildFake 200 px, LSUN/ddim 256 px), i.e. factor 1.

Per bucket, in train and val, the larger class is subsampled to the smaller
class's count; the excess goes to test (never wasted, never leaks).
Tampered (sid_tampered) -> test only, per protocol.

    python -m scripts.merge_ext [--only-sources sid_real,sid_fake,celebahq_1024]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

TAMPERED = {"sid_tampered"}
FIELDS = ["path", "orig", "label", "generator", "source", "long"]


def bucket(long_side: int) -> str:
    if long_side <= 341: return "<=341"
    if long_side <= 512: return "342-512"
    if long_side <= 768: return "513-768"
    if long_side <= 1024: return "769-1024"
    return ">1024"


def read(p):
    return list(csv.DictReader(open(p, newline="")))


def split(rows, fracs, seed):
    rows = rows[:]; random.Random(seed).shuffle(rows)
    n = len(rows); a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return rows[:a], rows[a:b], rows[b:]


def balance(rows, seed, tag):
    """Per bucket: keep min(n_real, n_fake) of each class; return kept, excess."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows: by[r["bucket"]][r["label"]].append(r)
    kept, excess = [], []
    rng = random.Random(seed)
    for b, cls in sorted(by.items()):
        n = min(len(cls.get("0", [])), len(cls.get("1", [])))
        for lab in ("0", "1"):
            rs = cls.get(lab, [])[:]; rng.shuffle(rs)
            kept += rs[:n]; excess += rs[n:]
        print(f"  {tag:5s} bucket {b:9s} real {len(cls.get('0', [])):6d} fake {len(cls.get('1', [])):6d} -> keep {n} each, {len(cls.get('0', []))+len(cls.get('1', []))-2*n} to test")
    return kept, excess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="data/manifests/canon3")
    ap.add_argument("--only-sources", default="", help="comma list; default all in canon_ext")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = {r["path"].removeprefix("./"): r for r in read("data/manifests/raw_ext.csv")}
    ext = []
    for r in read("data/manifests/canon_ext.csv"):
        m = raw[r["orig"].removeprefix("./")]
        r["long"] = str(max(int(m["w"]), int(m["h"]))); r["bucket"] = bucket(int(r["long"]))
        ext.append(r)
    if args.only_sources:
        keep = set(args.only_sources.split(","))
        ext = [r for r in ext if r["source"] in keep]

    out = {"train": [], "val": [], "test": []}
    for sp in out:
        for r in read(f"data/manifests/canon2_{sp}.csv"):
            r["long"] = "0"; r["bucket"] = "<=341"; out[sp].append(r)
    print(f"canon2: train {len(out['train'])} val {len(out['val'])} test {len(out['test'])}")

    by_src = defaultdict(list)
    for r in ext: by_src[r["source"]].append(r)
    e_tr, e_va, e_te = [], [], []
    for i, (src, rs) in enumerate(sorted(by_src.items())):
        if src in TAMPERED:
            e_te += rs; print(f"  {src:16s} n={len(rs):6d} -> TEST ONLY (tampered)"); continue
        a, b, c = split(rs, (0.8, 0.1), args.seed + i)
        e_tr += a; e_va += b; e_te += c
        print(f"  {src:16s} n={len(rs):6d} label={rs[0]['label']} long={sorted({r['long'] for r in rs})[:3]} -> {len(a)}/{len(b)}/{len(c)}")

    print("balance (ext rows only; canon2 is already balanced in its own bucket):")
    e_tr, x1 = balance(e_tr, args.seed + 100, "train")
    e_va, x2 = balance(e_va, args.seed + 101, "val")
    out["train"] += e_tr; out["val"] += e_va; out["test"] += e_te + x1 + x2

    for sp, rows in out.items():
        p = f"{args.out_prefix}_{sp}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        nr = sum(r["label"] == "0" for r in rows)
        print(f"{p}: {len(rows)} rows ({nr} real / {len(rows)-nr} fake)")


if __name__ == "__main__":
    main()

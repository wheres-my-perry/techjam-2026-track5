"""Gate: every native-size bucket must hold BOTH classes in ~equal amounts in
train and val (Thinh's rule 2026-08-29), else the shrink factor is a label.

    python -m scripts.bucket_audit --prefix data/manifests/canon3 [--strict]
FAIL if, in train or val, a bucket with >= 200 rows has one class missing or
a class ratio above --max-ratio (default 1.5). Test is reported only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from scripts.merge_ext import bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/manifests/canon3")
    ap.add_argument("--max-ratio", type=float, default=1.5)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    fail = False
    for sp in ("train", "val", "test"):
        c = defaultdict(lambda: [0, 0])
        for r in csv.DictReader(open(f"{args.prefix}_{sp}.csv", newline="")):
            c[bucket(int(r.get("long") or 0))][int(r["label"])] += 1
        print(f"== {sp}")
        for b, (nr, nf) in sorted(c.items()):
            ratio = max(nr, nf) / max(1, min(nr, nf))
            ok = (nr > 0 and nf > 0 and ratio <= args.max_ratio) or (nr + nf < 200)
            tag = "OK  " if ok else ("FAIL" if sp != "test" else "warn")
            if not ok and sp != "test": fail = True
            print(f"  {tag} bucket {b:9s} real {nr:7d} fake {nf:7d} ratio {ratio:.2f}")
    print("BUCKET AUDIT:", "FAIL" if fail else "CLEAN")
    if fail and args.strict: sys.exit(1)


if __name__ == "__main__":
    main()

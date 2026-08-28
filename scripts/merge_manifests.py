"""Merge canonicalized component manifests into canon2_{train,val,test}.csv.

Policy: existing WildFake canon splits stay in their splits; ArtiFact splits
80/10/10; LSUN reals split 60/10/30 (extra test share so 256-bucket reals
cover the ddpm holdout cells).
"""

from __future__ import annotations

import argparse
import csv
import random


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def split(rows, fracs, seed):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    a = int(n * fracs[0])
    b = a + int(n * fracs[1])
    return rows[:a], rows[a:b], rows[b:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="data/manifests/canon2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = read("data/manifests/canon_wf_train.csv")
    va = read("data/manifests/canon_wf_val.csv")
    te = read("data/manifests/canon_wf_test.csv")
    a_tr, a_va, a_te = split(read("data/manifests/canon_artifact.csv"),
                             (0.8, 0.1), args.seed)
    l_tr, l_va, l_te = split(read("data/manifests/canon_lsun.csv"),
                             (0.6, 0.1), args.seed + 1)
    outs = {"train": tr + a_tr + l_tr, "val": va + a_va + l_va,
            "test": te + a_te + l_te}
    for name, rows in outs.items():
        random.Random(args.seed + hash(name) % 1000).shuffle(rows)
        p = f"{args.out_prefix}_{name}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "label", "generator",
                                               "source"])
            w.writeheader()
            w.writerows(rows)
        n_real = sum(1 for r in rows if r["label"] == "0")
        print(f"{name}: {len(rows)} rows ({n_real} real / "
              f"{len(rows)-n_real} fake) -> {p}")


if __name__ == "__main__":
    main()

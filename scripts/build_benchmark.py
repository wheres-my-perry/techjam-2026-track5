"""Build the DECOUPLED benchmark: everything the training corpus does not cover.

Thinh's design: train on OmniFake, benchmark on what OmniFake lacks. The benchmark must share
nothing with training — not the generators, not the reals, not the preprocessing. So:

  * FAKES: only generators absent from the training corpus (compared name-wise, normalised).
  * REALS: our own reals, but hash-deduplicated against the training reals, because OmniFake's
    real half is drawn from the same public pools we hold (COCO / ImageNet / LAION / FFHQ). Without
    that, benchmark reals could be images the model trained on.
  * PATHS point at the ORIGINAL files, never the canonical crops, so the benchmark is scored
    through the production path (src/predict.py, vote(L=320)) and never through
    scripts/canonicalize.py -- applying the training transform to a benchmark manufactures
    agreement between them.

    python -m scripts.build_benchmark --test data/manifests/canon6_test.csv \
        --train data/manifests/omni_train.csv --out data/manifests/benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def norm(g):
    return g.replace("_", "").replace("-", "").lower()


def sha(path):
    try:
        with open(path, "rb") as fh:
            return path, hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/manifests/canon6_test.csv",
                    help="our held-out rows; `orig` is used, not `path`")
    ap.add_argument("--train", default="data/manifests/omni_train.csv")
    ap.add_argument("--out", default="data/manifests/benchmark.csv")
    ap.add_argument("--cap-per-generator", type=int, default=700)
    ap.add_argument("--cap-real", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-dedup", action="store_true")
    a = ap.parse_args()

    tr = list(csv.DictReader(open(a.train, newline="")))
    trained = {norm(r["generator"]) for r in tr if r["label"] == "1" and r["generator"]}
    trained |= {norm(r["generator"].replace("omni_", "")) for r in tr
                if r["label"] == "1" and r["generator"]}
    print(f"training generators: {len(trained)}")

    rows = list(csv.DictReader(open(a.test, newline="")))
    per, kept, skipped = Counter(), [], Counter()
    for r in rows:
        src_path = r.get("orig") or r["path"]
        if r["label"] == "1":
            g = norm(r["generator"])
            if not g:
                continue
            if g in trained:
                skipped[r["generator"]] += 1
                continue
            if per[r["generator"]] >= a.cap_per_generator:
                continue
            per[r["generator"]] += 1
        else:
            if per["__real__"] >= a.cap_real:
                continue
            per["__real__"] += 1
        kept.append({"path": src_path, "label": r["label"], "generator": r["generator"],
                     "source": r["source"], "long": r.get("long", "")})

    print(f"\nEXCLUDED (generator also in training): "
          f"{', '.join(f'{k} {v}' for k, v in skipped.most_common())}")
    gens = [k for k in per if k != '__real__']
    print(f"\nKEPT {len(gens)} disjoint generators:")
    for k in sorted(gens):
        print(f"   {k:26s} {per[k]}")
    print(f"   {'(reals)':26s} {per['__real__']}")

    if not a.no_dedup:
        tr_real = [r.get("orig") or r["path"] for r in tr if r["label"] == "0"]
        print(f"\nhashing {len(tr_real)} training reals + {per['__real__']} benchmark reals...",
              flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            trh = {h for _, h in ex.map(sha, tr_real, chunksize=64) if h}
            bench_real = [r["path"] for r in kept if r["label"] == "0"]
            bh = dict(ex.map(sha, bench_real, chunksize=64))
        drop = {p for p, h in bh.items() if h and h in trh}
        if drop:
            print(f"  DROPPED {len(drop)} benchmark reals that are byte-identical to a "
                  f"TRAINING real")
            kept = [r for r in kept if r["path"] not in drop]
        else:
            print("  no benchmark real is byte-identical to a training real")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source", "long"])
        w.writeheader(); w.writerows(kept)
    nr = sum(1 for r in kept if r["label"] == "0")
    print(f"\n{len(kept)} rows ({nr} real / {len(kept)-nr} fake) -> {a.out}")
    print("Paths are ORIGINAL files: score through the production path, never canonicalize.")


if __name__ == "__main__":
    main()

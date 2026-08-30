"""canon5 = canon4 with the WildFake label bug removed (found 2026-08-30 by the teammate's audit).

Bug: scripts/get_wildfake.py matched label-CSV rows to local files by FILENAME ONLY. WildFake's GAN
images (never downloaded) are named img000000.jpg... exactly like the real AFHQ/FFHQ photos that
were downloaded, so every "stylegan / vqvae / biggan / stargan" row in canon2..canon4 is a REAL
photograph labelled fake (train 47,981 rows = 24% of claimed fakes; val 5,968; test 6,046). Side
effects: 4,963 source files carried both labels, 4,555 files sat in both train and val.

Fix (manifest level, no re-canonicalisation needed):
  1. drop every wildfake row with label 1 and generator in {stylegan, vqvae, biggan, stargan}
  2. one row per source file in train (drop duplicates), and no val row whose source file is in
     train (those rows go to test, never wasted)
  3. per native-size bucket, in train and val, subsample the larger class to the smaller class's
     count (seeded); the excess goes to test — Thinh's balance rule, same as merge_ext
  4. test keeps everything else (still labelled by source; the bogus GAN rows are dropped there too)

    python -m scripts.fix_canon5 [--in-prefix data/manifests/canon4] [--out-prefix data/manifests/canon5]
Then run the gates: bucket_audit, shortcut_audit, canary_audit on canon5_train.
"""
from __future__ import annotations
import argparse, csv, random
from collections import defaultdict, Counter

BAD_GEN = {"stylegan", "vqvae", "biggan", "stargan"}
FIELDS = ["path", "orig", "label", "generator", "source", "long"]


def bucket(long_side: int) -> str:
    if long_side <= 341: return "<=341"
    if long_side <= 640: return "<=640"
    return ">640"


def read(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))


def write(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def is_bogus(r):
    return r["label"] == "1" and r["source"] == "wildfake" and r["generator"] in BAD_GEN


def balance(rows, rng, name):
    by = defaultdict(lambda: {"0": [], "1": []})
    for r in rows: by[bucket(int(r["long"]))][r["label"]].append(r)
    keep, excess = [], []
    for b, d in sorted(by.items()):
        n = min(len(d["0"]), len(d["1"]))
        for lab in ("0", "1"):
            rs = d[lab][:]; rng.shuffle(rs)
            keep += rs[:n]; excess += rs[n:]
        print(f"  {name} bucket {b:6s}: real {len(d['0']):6d} fake {len(d['1']):6d} -> {n} each, {len(d['0']) + len(d['1']) - 2 * n} to test")
    return keep, excess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-prefix", default="data/manifests/canon4")
    ap.add_argument("--out-prefix", default="data/manifests/canon5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tr, va, te = (read(f"{a.in_prefix}_{s}.csv") for s in ("train", "val", "test"))
    n0 = (len(tr), len(va), len(te))
    tr = [r for r in tr if not is_bogus(r)]; va = [r for r in va if not is_bogus(r)]; te = [r for r in te if not is_bogus(r)]
    print(f"dropped bogus GAN rows: train {n0[0] - len(tr)}, val {n0[1] - len(va)}, test {n0[2] - len(te)}")
    seen, dedup = set(), []
    for r in tr:
        if r["orig"] in seen: continue
        seen.add(r["orig"]); dedup.append(r)
    print(f"train duplicate rows removed: {len(tr) - len(dedup)}"); tr = dedup
    leak = [r for r in va if r["orig"] in seen]; va = [r for r in va if r["orig"] not in seen]
    print(f"val rows whose source file is in train -> test: {len(leak)}"); te += leak
    tr, ex1 = balance(tr, rng, "train"); va, ex2 = balance(va, rng, "val"); te += ex1 + ex2
    dev = {r["orig"] for r in tr} | {r["orig"] for r in va}
    n = len(te); te = [r for r in te if r["orig"] not in dev]
    print(f"test rows whose source file is in train/val dropped: {n - len(te)}")
    seen_t, dd = set(), []
    for r in te:
        if r["orig"] in seen_t: continue
        seen_t.add(r["orig"]); dd.append(r)
    print(f"test duplicate rows removed: {len(te) - len(dd)}"); te = dd
    lab = defaultdict(set)
    for r in tr + va: lab[r["orig"]].add(r["label"])
    assert not any(len(v) > 1 for v in lab.values()), "label conflict remains"
    assert not ({r["orig"] for r in tr} & {r["orig"] for r in va}), "train/val overlap remains"
    for s, rows in (("train", tr), ("val", va), ("test", te)):
        write(f"{a.out_prefix}_{s}.csv", rows)
        c = Counter(r["label"] for r in rows)
        print(f"{a.out_prefix}_{s}.csv: {len(rows)} rows (real {c['0']}, fake {c['1']})")


if __name__ == "__main__":
    main()

"""Build the ONE partial-edit evaluation set that is unseen by BOTH models.

canon6_mlp trained on canon6_train (no partial edits at all).
canon6pe_mlp trains on canon6pe_train (80% of the partial edits).
So the eval set may only contain rows that are in NEITHER training file:
  fakes = partial edits sitting in canon6pe_test
  reals = reals in canon6pe_test AND canon6_test
Matched 1:1 per native-size bucket so a size confound cannot create the number.
"""
import csv, random, collections, sys

PARTIAL_EDIT = {"sid_tampered", "lama", "mat", "generative_inpainting", "palette"}
def bucket(l):
    return "<=341" if l <= 341 else "342-512" if l <= 512 else "513-768" if l <= 768 else "769-1024" if l <= 1024 else ">1024"

def rows(p):
    return list(csv.DictReader(open(p, newline="")))

M = "data/manifests/"
tr_a = {r["orig"] for r in rows(M + "canon6_train.csv")} | {r["orig"] for r in rows(M + "canon6_val.csv")}
tr_b = {r["orig"] for r in rows(M + "canon6pe_train.csv")} | {r["orig"] for r in rows(M + "canon6pe_val.csv")}
seen = tr_a | tr_b

pe_test = rows(M + "canon6pe_test.csv")
c6_test = {r["orig"] for r in rows(M + "canon6_test.csv")}

fakes = [r for r in pe_test if r["generator"] in PARTIAL_EDIT and r["orig"] not in seen]
reals = [r for r in pe_test if r["label"] == "0" and r["orig"] in c6_test and r["orig"] not in seen]
print(f"candidate fakes {len(fakes)}  candidate reals {len(reals)}")

by_f = collections.defaultdict(list); by_r = collections.defaultdict(list)
for r in fakes: by_f[bucket(int(r["long"]))].append(r)
for r in reals: by_r[bucket(int(r["long"]))].append(r)

out = []
for b in sorted(set(by_f) | set(by_r)):
    rng = random.Random(hash(b) & 0xFFFF)
    n = min(len(by_f[b]), len(by_r[b]))
    print(f"  {b:10s} fake {len(by_f[b]):5d} real {len(by_r[b]):5d} -> keep {n} each")
    if n == 0: continue
    out += rng.sample(by_f[b], n) + rng.sample(by_r[b], n)

random.Random(0).shuffle(out)
dst = M + "edits_eval.csv"
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["path", "orig", "label", "generator", "source", "long"], extrasaction="ignore")
    w.writeheader(); w.writerows(out)
nr = sum(r["label"] == "0" for r in out)
print(f"{dst}: {len(out)} rows ({nr} real / {len(out)-nr} partial-edit fake)")
gens = collections.Counter(r["generator"] for r in out if r["label"] == "1")
print("  generators:", dict(gens))
assert not ({r["orig"] for r in out} & seen), "LEAK: eval row seen in a training manifest"
print("LEAK CHECK PASS: no eval row appears in either model's train/val")

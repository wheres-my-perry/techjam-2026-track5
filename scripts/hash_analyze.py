"""Fast analysis of scripts/hash_audit.py's CSV (sha256 + dHash per file across all groups).

Exact-byte duplicates within/across groups + label conflicts, and perceptual near-duplicates
(dHash Hamming <= MAXD) between canon5_train and every other group, using pigeonhole bucketing:
split the 64-bit hash into MAXD+1 pieces; two hashes within Hamming MAXD share at least one piece.

    python -m scripts.hash_analyze --csv outputs/audit/hash_audit.csv [--maxd 4] [--out outputs/audit/hash_audit.txt]
"""
from __future__ import annotations
import argparse, csv, os
from collections import Counter, defaultdict


def pieces(h: int, k: int):
    w = 64 // k
    return [((h >> (w * i)) & ((1 << w) - 1), i) for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/audit/hash_audit.csv")
    ap.add_argument("--maxd", type=int, default=4)
    ap.add_argument("--out", default="outputs/audit/hash_audit.txt")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv, newline=""))]
    ok = [r for r in rows if r["sha256"] != "ERR"]
    out = [f"files hashed: {len(rows)}  (errors: {len(rows) - len(ok)})",
           "groups: " + ", ".join(f"{g}={n}" for g, n in Counter(r["group"] for r in ok).items())]
    by_sha = defaultdict(list)
    for r in ok: by_sha[r["sha256"]].append(r)
    within, across, conflict = Counter(), Counter(), []
    for rs in by_sha.values():
        if len(rs) < 2: continue
        gs = Counter(r["group"] for r in rs)
        for g, n in gs.items():
            if n > 1: within[g] += n - 1
        if len(gs) > 1: across[" + ".join(sorted(gs))] += 1
        if len({r["label"] for r in rs}) > 1: conflict.append(rs)
    out.append("\nEXACT-BYTE duplicates within a group (extra copies): " + (", ".join(f"{g}: {n}" for g, n in within.items()) or "none"))
    out.append("EXACT-BYTE files shared ACROSS groups: " + (", ".join(f"[{k}]: {n}" for k, n in across.items()) or "none"))
    out.append(f"EXACT-BYTE label conflicts (same bytes, different labels): {len(conflict)}")
    for rs in conflict[:5]: out.append("   e.g. " + " | ".join(f"{r['group']}:{r['label']}:{r['src']}:{os.path.basename(r['path'])}" for r in rs))
    train = [r for r in ok if r["group"] == "canon5_train"]
    k = a.maxd + 1
    idx = defaultdict(list)
    for i, r in enumerate(train):
        for p in pieces(int(r["dhash"], 16), k): idx[p].append(i)
    def near(group):
        q = [r for r in ok if r["group"] == group]
        hits, ex, n_hit = Counter(), [], 0
        for r in q:
            h = int(r["dhash"], 16); best = None
            cand = set()
            for p in pieces(h, k): cand.update(idx.get(p, ()))
            for i in cand:
                d = bin(h ^ int(train[i]["dhash"], 16)).count("1")
                if d <= a.maxd and (best is None or d < best[0]): best = (d, i)
            if best is not None:
                n_hit += 1; hits[r["src"]] += 1
                if len(ex) < 4:
                    t = train[best[1]]
                    ex.append(f"{os.path.basename(r['path'])} (label {r['label']}, {r['src']}) ~ train {os.path.basename(t['path'])} (label {t['label']}, {t['src']}) d={best[0]}")
        return n_hit, len(q), hits, ex
    for g in ("official", "unseen", "wild", "canon5_val", "canon5_test"):
        n_hit, n, hits, ex = near(g)
        if n == 0: continue
        out.append(f"\nPERCEPTUAL near-duplicates (dHash Hamming<={a.maxd}) of {g} images in canon5_train: {n_hit}/{n}")
        for kk, v in hits.most_common(8): out.append(f"   {v:5d}  {kk}")
        for e in ex: out.append("   e.g. " + e)
    report = "\n".join(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()

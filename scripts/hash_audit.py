"""Byte-level and perceptual duplicate audit across ALL sets we train on or report on.

Computes, per image: sha256 of the file bytes and a 64-bit dHash (perceptual, on the decoded
image), for:
  - canon5 train / val / test  (the canonical `path` files)          -> group canon5_<split>
  - the DALL-E benchmark manifest (data/manifests/canon_official.csv) -> group official
  - the unseen-64 set (scratchpad randtest folders)                   -> group unseen
  - the wild set (data/hack)                                          -> group wild
Then reports: exact-byte duplicates within and across groups (with labels), label conflicts by
bytes, and perceptual near-duplicates (dHash Hamming <= 4) between every training image and
every benchmark image (official, unseen, wild). Outputs a CSV of hashes and a text report.

    python -m scripts.hash_audit --out outputs/audit/hash_audit --workers 24 [--unseen DIR]
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageOps


def dhash(im, size=8):
    g = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def one(args):
    path, group, label, src = args
    try:
        with open(path, "rb") as f: b = f.read()
        sha = hashlib.sha256(b).hexdigest()
        with Image.open(path) as im: dh = dhash(im)
        return (path, group, label, src, sha, f"{dh:016x}")
    except Exception as e:
        return (path, group, label, src, "ERR", "ERR")


def rows_from_manifest(p, group):
    with open(p, newline="") as f:
        return [(r["path"], group, r["label"], r.get("source", "") or r.get("generator", "")) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/manifests/canon5")
    ap.add_argument("--official", default="data/manifests/canon_official.csv")
    ap.add_argument("--unseen", default="")
    ap.add_argument("--wild", default="data/hack")
    ap.add_argument("--out", default="outputs/audit/hash_audit")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    items = []
    for sp in ("train", "val", "test"):
        items += rows_from_manifest(f"{a.prefix}_{sp}.csv", f"canon5_{sp}")
    items += rows_from_manifest(a.official, "official")
    if a.unseen:
        for d in sorted(glob.glob(os.path.join(a.unseen, "*"))):
            if os.path.isdir(d):
                lab = "0" if os.path.basename(d).startswith("real_") else "1"
                items += [(os.path.realpath(p), "unseen", lab, os.path.basename(d)) for p in sorted(glob.glob(d + "/*"))]
    if os.path.isdir(a.wild):
        for g, lab in (("real", "0"), ("gemini", "1")):
            items += [(p, "wild", lab, g) for p in sorted(glob.glob(os.path.join(a.wild, g, "*")))]
    print(f"hashing {len(items)} files with {a.workers} workers", flush=True)
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, items, chunksize=256)):
            res.append(r)
            if i % 50000 == 0: print(f"  {i}/{len(items)}", flush=True)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "group", "label", "src", "sha256", "dhash"]); w.writerows(res)
    err = sum(r[4] == "ERR" for r in res)
    out = [f"files hashed: {len(res)}  (errors: {err})"]
    by_sha = defaultdict(list)
    for r in res:
        if r[4] != "ERR": by_sha[r[4]].append(r)
    groups = sorted({r[1] for r in res})
    # exact duplicates: within group, across groups, label conflicts
    within, across, conflict = Counter(), Counter(), []
    for sha, rs in by_sha.items():
        if len(rs) < 2: continue
        gs = Counter(r[1] for r in rs)
        for g, n in gs.items():
            if n > 1: within[g] += n - 1
        if len(gs) > 1: across[tuple(sorted(gs))] += 1
        if len({r[2] for r in rs}) > 1: conflict.append(rs)
    out.append("\nEXACT-BYTE duplicates within a group (extra copies): " + (", ".join(f"{g}: {n}" for g, n in within.items()) or "none"))
    out.append("EXACT-BYTE files shared ACROSS groups: " + (", ".join(f"{'+'.join(k)}: {n}" for k, n in across.items()) or "none"))
    out.append(f"EXACT-BYTE label conflicts (same bytes, different labels): {len(conflict)}")
    for rs in conflict[:5]: out.append("   e.g. " + " | ".join(f"{r[1]}:{r[2]}:{r[3]}:{os.path.basename(r[0])}" for r in rs))
    # perceptual near-duplicates: train vs each benchmark, and val/test vs train
    train = [r for r in res if r[1] == "canon5_train" and r[5] != "ERR"]
    tr_hash = np.array([int(r[5], 16) for r in train], dtype=np.uint64)
    def near(group, maxd=4):
        q = [r for r in res if r[1] == group and r[5] != "ERR"]
        if not q: return None
        qh = np.array([int(r[5], 16) for r in q], dtype=np.uint64)
        hits = Counter(); n_hit = 0; ex = []
        for i in range(0, len(qh), 512):
            x = np.bitwise_xor(qh[i:i + 512, None], tr_hash[None, :])
            d = np.zeros(x.shape, dtype=np.uint8)
            for k in range(8): d += np.unpackbits((x >> np.uint64(8 * k)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
            m = d <= maxd
            for j in np.flatnonzero(m.any(1)):
                n_hit += 1; hits[q[i + j][3]] += 1
                if len(ex) < 3:
                    t = train[int(np.flatnonzero(m[j])[0])]
                    ex.append(f"{os.path.basename(q[i + j][0])} (label {q[i + j][2]}) ~ train {os.path.basename(t[0])} (label {t[2]}, {t[3]})")
        return n_hit, len(q), hits, ex
    for g in ("official", "unseen", "wild", "canon5_val", "canon5_test"):
        r = near(g)
        if r is None: continue
        n_hit, n, hits, ex = r
        out.append(f"\nPERCEPTUAL near-duplicates (dHash Hamming<=4) of {g} images in canon5_train: {n_hit}/{n}")
        for k, v in hits.most_common(8): out.append(f"   {v:5d}  {k}")
        for e in ex: out.append("   e.g. " + e)
    report = "\n".join(out)
    with open(a.out + ".txt", "w") as f: f.write(report + "\n")
    print(report)
    print("HASH_AUDIT_DONE")


if __name__ == "__main__":
    main()

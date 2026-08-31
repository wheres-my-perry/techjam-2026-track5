"""Full state of the training data, per native-size bucket. Regenerate after EVERY change.

Thinh's standing order (AGENTS.md 1.1): after every modification to the training set, look at the
distribution again. This prints, per native-size bucket: how many reals and fakes, WHICH real
sources and WHICH fake generators, and the subject mix on each side -- because the classes can be
perfectly balanced in COUNT inside a bucket and still be disjoint in CONTENT (canon6's 342-512
bucket was 10/12 animal close-ups on the real side and diverse commercial scenes on the fake side,
and passed every gate).

    python -m scripts.data_report --prefix data/manifests/canon6 --md docs/DATA_STATE.md
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

ORDER = ["<=341", "342-512", "513-768", "769-1024", ">1024"]
SPLITS = ("train", "val", "test")


def buck(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data/manifests/canon6")
    ap.add_argument("--md", default=None)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    try:
        from scripts.content_audit import load_artifact_categories, subject
        cats = load_artifact_categories()
    except Exception:
        cats, subject = {}, None

    L = []
    def out(s=""):
        print(s)
        L.append(s)

    man = {}
    for sp in SPLITS:
        p = f"{a.prefix}_{sp}.csv"
        if os.path.exists(p):
            man[sp] = list(csv.DictReader(open(p, newline="")))

    out(f"# Data state — `{a.prefix}`")
    out()
    out("| split | total | real | fake | fake generators |")
    out("|---|---|---|---|---|")
    for sp, rows in man.items():
        nr = sum(1 for r in rows if r["label"] == "0")
        gens = len({r["generator"] for r in rows if r["label"] == "1" and r["generator"]})
        out(f"| **{sp}** | {len(rows):,} | {nr:,} | {len(rows)-nr:,} | {gens} |")
    if "train" in man and "val" in man:
        n = len(man["train"]) + len(man["val"])
        out()
        out(f"**Images actually pushed into training (train + val): {n:,}** "
            f"({len(man['train']):,} train / {len(man['val']):,} val). "
            f"Test is never trained on.")

    for sp, rows in man.items():
        out()
        out(f"## {sp} — by native-size bucket")
        by = defaultdict(lambda: [0, 0])
        src = defaultdict(lambda: defaultdict(Counter))
        subj = defaultdict(lambda: defaultdict(Counter))
        for r in rows:
            if not r.get("long"):
                continue
            b = buck(r["long"]); lab = int(r["label"])
            by[b][lab] += 1
            src[b][lab][r["source"] if lab == 0 else (r["generator"] or "?")] += 1
            if subject:
                subj[b][lab][subject(r, cats)] += 1
        for b in ORDER:
            if b not in by:
                continue
            nr, nf = by[b]
            ratio = f"{nr/nf:.2f}:1" if nf else "no fakes"
            out()
            out(f"### {b} px — {nr + nf:,} images · {nr:,} real / {nf:,} fake ({ratio})")
            out()
            out(f"| | sources / generators | subjects |")
            out("|---|---|---|")
            for lab, tag in ((0, "REAL"), (1, "AI")):
                s = ", ".join(f"{k} {v:,}" for k, v in src[b][lab].most_common(a.top)) or "—"
                t = ", ".join(f"{k} {v:,}" for k, v in subj[b][lab].most_common(4)) if subject else "—"
                out(f"| **{tag}** | {s} | {t} |")

    if "train" in man:
        out()
        out("## train — every source and generator")
        out()
        rs = Counter(r["source"] for r in man["train"] if r["label"] == "0")
        gs = Counter(r["generator"] for r in man["train"] if r["label"] == "1")
        out(f"**{len(rs)} real sources** ({sum(rs.values()):,} images)")
        out()
        out("| real source | n |")
        out("|---|---|")
        for k, v in rs.most_common():
            out(f"| {k} | {v:,} |")
        out()
        out(f"**{len(gs)} fake generators** ({sum(gs.values()):,} images)")
        out()
        out("| generator | n |")
        out("|---|---|")
        for k, v in gs.most_common():
            out(f"| {k} | {v:,} |")

    if "test" in man:
        tr_g = {r["generator"] for r in man.get("train", []) if r["label"] == "1"}
        te_g = {r["generator"] for r in man["test"] if r["label"] == "1"}
        held = sorted(g for g in te_g - tr_g if g)
        out()
        out("## Held out of training entirely (test only)")
        out()
        c = Counter(r["generator"] for r in man["test"] if r["generator"] in held)
        out("| generator | n in test |")
        out("|---|---|")
        for g in held:
            out(f"| {g} | {c[g]:,} |")

    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w").write("\n".join(L) + "\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()

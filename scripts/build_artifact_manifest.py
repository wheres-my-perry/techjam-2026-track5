"""Build a manifest from the extracted ArtiFact tree.

    python -m scripts.build_artifact_manifest --root data/artifact \
        --cap-real 150000 --cap-fake 150000

LABEL SOURCE (fixed 2026-08-28): the per-image `target` column in each
folder's metadata.csv. target==0 -> real, anything else -> fake. Verified
against the approved plan: this reproduces ArtiFact's published totals
exactly (964,989 real / 1,531,749 fake / 2,496,738 files).

Do NOT label by folder name. The previous version did, and it was wrong in
three separate ways:
  * the tree is <root>/ArtiFact/{Real,Fake}/<source>/, so a name-based scan
    of <root> saw ONE folder ("ArtiFact"), matched no real name, and marked
    all 2.5M files fake -- 36.8% of the sampled "fakes" were real photos;
  * the Real/ vs Fake/ parent is not reliable either: Fake/afhq is 31,933
    REAL photos, and Real/cycle_gan is half fake;
  * two folders (pro_gan, cycle_gan) hold BOTH classes, so no folder-level
    label can ever be correct for them.

Sampling is spread across sources (water-filling) rather than uniform over
files: stylegan2 alone has 1M images and would otherwise be ~65% of the fake
half, collapsing generator diversity.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

# Sources whose images are real photographs, for the printed cross-check only.
# The `target` column is what actually decides the label.
EXPECTED_REAL = {"afhq", "celebahq", "coco", "cycle_gan", "ffhq", "imagenet",
                 "landscape", "lsun", "metfaces"}


def find_root(root: str) -> str:
    """Accept either <root> or <root>/ArtiFact as the tree top."""
    for cand in (os.path.join(root, "ArtiFact"), root):
        if os.path.isdir(os.path.join(cand, "Real")) or \
           os.path.isdir(os.path.join(cand, "Fake")):
            return cand
    raise SystemExit(f"no Real/ or Fake/ under {root} -- is ArtiFact extracted?")


def allocate(avail: dict[str, int], cap: int) -> dict[str, int]:
    """Spread `cap` picks across sources; small sources give all they have."""
    out, remaining = {}, cap
    order = sorted(avail, key=lambda k: avail[k])
    for i, k in enumerate(order):
        share = remaining // (len(order) - i)
        out[k] = min(avail[k], share)
        remaining -= out[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/artifact")
    ap.add_argument("--out", default="data/manifests/artifact_raw.csv")
    ap.add_argument("--cap-real", type=int, default=150000)
    ap.add_argument("--cap-fake", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    top = find_root(args.root)
    pools: dict[tuple[str, int], list[dict]] = {}
    print(f"{'folder':32s} {'real':>9s} {'fake':>9s}  note", flush=True)
    mismatches = []
    for parent in ("Real", "Fake"):
        pdir = os.path.join(top, parent)
        if not os.path.isdir(pdir):
            continue
        for src in sorted(os.listdir(pdir)):
            md = os.path.join(pdir, src, "metadata.csv")
            if not os.path.isfile(md):
                continue
            n = {0: 0, 1: 0}
            with open(md, newline="") as fh:
                for row in csv.DictReader(fh):
                    label = 0 if row["target"].strip() == "0" else 1
                    p = os.path.join(pdir, src, row["image_path"])
                    pools.setdefault((src, label), []).append(
                        {"path": p, "label": label,
                         "generator": "" if label == 0 else src,
                         "source": f"artifact_{src}" if label == 0
                                   else "artifact"})
                    n[label] += 1
            note = ""
            if n[0] and n[1]:
                note = "MIXED — both classes in one folder"
            elif (n[0] > 0) != (src in EXPECTED_REAL):
                note = "UNEXPECTED — label disagrees with source list"
            if (n[0] > 0) != (parent == "Real"):
                note = (note + "; " if note else "") + \
                       f"folder says {parent}, target says " \
                       f"{'real' if n[0] else 'fake'}"
            if note:
                mismatches.append(f"{parent}/{src}: {note}")
            print(f"{parent + '/' + src:32s} {n[0]:9d} {n[1]:9d}  {note}",
                  flush=True)

    if mismatches:
        print("\nFOLDER/LABEL MISMATCHES (labels come from target, so these "
              "are handled correctly — listed for the record):")
        for m in mismatches:
            print(f"  {m}")

    rng = random.Random(args.seed)
    rows = []
    for label, cap in ((0, args.cap_real), (1, args.cap_fake)):
        avail = {s: len(v) for (s, l), v in pools.items() if l == label}
        take = allocate(avail, cap)
        print(f"\n{'real' if label == 0 else 'fake'} allocation "
              f"(cap {cap}, {len(avail)} sources):", flush=True)
        for s in sorted(take, key=lambda k: -take[k]):
            if take[s]:
                print(f"  {s:28s} {take[s]:7d} / {avail[s]:8d}")
        for s, k in take.items():
            pool = pools[(s, label)]
            rng.shuffle(pool)
            rows += pool[:k]

    rng.shuffle(rows)
    n_real = sum(1 for r in rows if r["label"] == 0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator",
                                           "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{n_real} real + {len(rows) - n_real} fake -> {args.out}")


if __name__ == "__main__":
    main()

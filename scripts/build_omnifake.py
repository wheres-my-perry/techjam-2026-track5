"""Independent side test: OmniFake (OmniDFA paper, 1.17M images / 45 generators).

Thinh (2026-08-31): use this "for side testing (checking if the model is actually working and not
corrupted by the data bugs we had earlier)".

Every benchmark we build shares construction, canonicalization and sources with our training data,
so a good score there can reward our pipeline rather than detection ability — this corpus has
already produced four content/size confounds that all passed gates. OmniFake was built by other
people, from other generators, with other preprocessing, so it cannot share those specific flaws.

EVAL ONLY. Never enters training.

Stated limitation: only OmniFake's single-zip generators are used (its `real` half is a 41-part
217 GB archive), so the FAKE side is fully independent while the REAL side comes from our own
never-trained sources. That tests generalization to foreign generators, not a fully foreign
benchmark, and must be described that way.

    python -m scripts.build_omnifake --root data/omnifake/data --out data/manifests/raw_omnifake.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# OmniFake generator -> whether the NAME also appears in canon6 training. Same name from a
# different pipeline is still a different render, but the distinction must be reported.
ALSO_IN_TRAIN = {"DDIM", "GLIDE", "VQVAE", "StyleGAN_3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/omnifake/data")
    ap.add_argument("--out", default="data/manifests/raw_omnifake.csv")
    ap.add_argument("--cap", type=int, default=1500, help="images per generator")
    a = ap.parse_args()

    rows, per = [], Counter()
    for d in sorted(os.listdir(a.root)):
        if not d.startswith("x_"):
            continue
        gen = d[2:]
        base = os.path.join(a.root, d)
        n = 0
        for dp, _, fns in os.walk(base):
            for fn in sorted(fns):
                if n >= a.cap:
                    break
                if os.path.splitext(fn)[1].lower() not in IMG:
                    continue
                p = os.path.join(dp, fn)
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    continue
                rows.append({"path": p, "label": 1, "generator": f"omni_{gen.lower()}",
                             "source": "omnifake", "w": w, "h": h})
                n += 1
            if n >= a.cap:
                break
        per[gen] = n
        tag = "  (name also in canon6 train)" if gen in ALSO_IN_TRAIN else ""
        print(f"  {gen:16s} {n:6d} images{tag}", flush=True)

    empty = [g for g, n in per.items() if n == 0]
    if empty:
        print(f"!! {len(empty)} generator(s) produced NO images: {empty}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source", "w", "h"])
        w_.writeheader(); w_.writerows(rows)
    sizes = Counter(max(r["w"], r["h"]) for r in rows)
    print(f"\n{len(rows)} fakes from {sum(1 for v in per.values() if v)} generators -> {a.out}")
    print("native long side:", sizes.most_common(6))
    print("\nNOTE: fakes only. Pair with never-trained reals at eval time; the real half is ours.")


if __name__ == "__main__":
    main()

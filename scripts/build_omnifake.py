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

def trained_generators(manifest="data/manifests/canon6_train.csv"):
    """Generator names actually present in TRAINING, read from the manifest.

    This was a hardcoded guess and it was wrong: it listed DDIM and VQVAE as trained when ddim is
    HELD OUT (a decision made the same day) and we train vq_diffusion, a different model. That
    halved the count of genuinely-unseen generators in our own result. Never assert the contents
    of a manifest from memory -- read it.
    """
    import csv as _csv
    import os as _os
    if not _os.path.exists(manifest):
        print(f"!! {manifest} missing — cannot say which generators are trained")
        return None
    with open(manifest, newline="") as fh:
        return {r["generator"].replace("_", "").lower()
                for r in _csv.DictReader(fh) if r["label"] == "1" and r["generator"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/omnifake/data")
    ap.add_argument("--out", default="data/manifests/raw_omnifake.csv")
    ap.add_argument("--cap", type=int, default=1500, help="images per generator")
    a = ap.parse_args()

    trained = trained_generators()
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
        if trained is None:
            tag = "  (training set unknown)"
        elif gen.replace("_", "").lower() in trained:
            tag = "  <- this generator IS in our training set (easier test)"
        else:
            tag = "  <- NOT in our training set (true unseen-generator test)"
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

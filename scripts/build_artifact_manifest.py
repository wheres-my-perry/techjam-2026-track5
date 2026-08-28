"""Build a manifest from the extracted ArtiFact tree (real/fake by folder name).

    python -m scripts.build_artifact_manifest --root data/artifact \
        --cap-real 150000 --cap-fake 150000

Prints the per-folder classification so the log shows the tree structure;
UNKNOWN folder names are treated as fake generators (ArtiFact's layout is one
folder per source/generator) — verify the printed table before trusting the
manifest, and the audit gates run afterwards regardless.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

REAL_NAMES = {"afhq", "celebahq", "celeba_hq", "celeba-hq", "coco", "ffhq",
              "landscape", "lsun", "metfaces", "sfhq"}
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/artifact")
    ap.add_argument("--out", default="data/manifests/artifact_raw.csv")
    ap.add_argument("--cap-real", type=int, default=150000)
    ap.add_argument("--cap-fake", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reals, fakes = [], []
    for top in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, top)
        if not os.path.isdir(d):
            continue
        files = []
        for dirpath, _, names in os.walk(d):
            files += [os.path.join(dirpath, n) for n in names
                      if n.lower().endswith(EXTS)]
        is_real = top.lower().replace("-", "_") in {r.replace("-", "_")
                                                    for r in REAL_NAMES}
        kind = "REAL" if is_real else "FAKE"
        print(f"{kind:5s} {top:24s} {len(files):7d} files", flush=True)
        for p in files:
            if is_real:
                reals.append({"path": p, "label": 0, "generator": "",
                              "source": f"artifact_{top}"})
            else:
                fakes.append({"path": p, "label": 1, "generator": top,
                              "source": "artifact"})

    rng = random.Random(args.seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    rows = reals[: args.cap_real] + fakes[: args.cap_fake]
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{min(len(reals), args.cap_real)} real + "
          f"{min(len(fakes), args.cap_fake)} fake -> {args.out}")


if __name__ == "__main__":
    main()

"""Image-size distribution per class/generator for a manifest — shortcut audit.

    python -m scripts.size_audit --manifest data/manifests/wildfake_test.csv

If real and fake rows show disjoint size sets, size is a label giveaway and
AUROC measured on that manifest is inflated (2026-08-28 official_val lesson).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from src.data import load_image, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args()
    s = load_manifest(args.manifest)
    random.Random(0).shuffle(s)
    s = s[: args.limit]
    groups: dict[str, Counter] = {}
    for x in s:
        key = "real" if x.label == 0 else (x.generator or "fake")
        try:
            groups.setdefault(key, Counter())[load_image(x.path).size] += 1
        except Exception as e:
            print(f"skip {x.path}: {e}", flush=True)
    for k in sorted(groups):
        top = ", ".join(f"{w}x{h}:{n}" for (w, h), n in groups[k].most_common(4))
        print(f"{k:16s} n={sum(groups[k].values()):5d}  {top}", flush=True)


if __name__ == "__main__":
    main()

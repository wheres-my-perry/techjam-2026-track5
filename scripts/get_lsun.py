"""Stream LSUN reals from HF into local PNGs + manifest.

    python -m scripts.get_lsun --count 15000 --out data/lsun_church
    python -m scripts.get_lsun --dataset pcuenq/lsun-bedrooms --count 25000 \
        --out data/lsun_bedroom --manifest data/manifests/lsun_bedroom_raw.csv \
        --source lsun_bedroom

Both HF sets are the same LSUN pipeline (256 short side, same JPEG chain),
so church and bedroom are metadata-identical: the 256-bucket reals that pair
with WildFake's 256px ddim/ddpm fakes, whose content is exactly church +
bedroom (+ a general-photo set, CC9K).

Resume-safe: existing files are not re-saved (enumeration is seed-stable).
"""

from __future__ import annotations

import argparse
import csv
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=45000)
    ap.add_argument("--out", default="data/lsun_church")
    ap.add_argument("--manifest", default="data/manifests/lsun_raw.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="tglcourse/lsun_church_train")
    ap.add_argument("--source", default="lsun_church",
                    help="source tag written to the manifest")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train",
                      streaming=True).shuffle(seed=args.seed, buffer_size=10000)
    os.makedirs(args.out, exist_ok=True)
    rows, t0 = [], time.time()
    for i, ex in enumerate(ds):
        if i >= args.count:
            break
        p = os.path.join(args.out, f"lsun_{i:06d}.png")
        if not os.path.exists(p):
            img = ex["image"]
            tmp = p + ".tmp.png"
            img.convert("RGB").save(tmp, format="PNG")
            os.replace(tmp, p)
        rows.append({"path": p, "label": 0, "generator": "",
                     "source": args.source})
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"{i+1}/{args.count} ({rate:.0f}/s, "
                  f"eta {(args.count-i-1)/max(rate,1e-9)/60:.0f}m)", flush=True)

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.manifest}")


if __name__ == "__main__":
    main()

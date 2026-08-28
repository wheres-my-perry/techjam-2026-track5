"""Compression-history stress sets (shortcut hunt, 2026-08-29).

Every real in canon2 has been through JPEG at least once (camera/web) and
ArtiFact re-JPEGs both classes, while diffusion fakes are born as PNG. So
"fewer JPEG generations = fake" is a dataset-wide rule. Test whether the
model reads it: give the FAKES one extra JPEG pass (history now matches the
reals') and see if the held-out ddpm AUROC survives.

    fakejpeg : fakes JPEG'd once (q 75-95), reals untouched   <- the test
    realjpeg : reals JPEG'd once, fakes untouched             <- mirror
    bothjpeg : both JPEG'd once                               <- control

All outputs re-saved as PNG so file format stays uniform (metadata audit).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random

from PIL import Image

from src.data import load_manifest


def jpeg_once(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, subsampling=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/canon2_test.csv")
    ap.add_argument("--n", type=int, default=1500, help="per class")
    ap.add_argument("--fake-gen", default="ddpm")
    ap.add_argument("--out-dir", default="data/stress")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    s = load_manifest(args.manifest)
    reals = [x for x in s if x.label == 0]
    fakes = [x for x in s if x.label == 1 and x.generator == args.fake_gen]
    rng.shuffle(reals); rng.shuffle(fakes)
    reals, fakes = reals[: args.n], fakes[: args.n]
    for variant, jr, jf in (("fakejpeg", False, True), ("realjpeg", True, False),
                            ("bothjpeg", True, True)):
        d = os.path.join(args.out_dir, variant)
        os.makedirs(d, exist_ok=True)
        rows = []
        for lab, group, do in ((0, reals, jr), (1, fakes, jf)):
            for x in group:
                im = Image.open(x.path).convert("RGB")
                if do:
                    im = jpeg_once(im, rng.randint(75, 95))
                p = os.path.join(d, os.path.basename(x.path))
                if not os.path.exists(p):
                    im.save(p, format="PNG")
                rows.append({"path": p, "orig": x.path, "label": lab,
                             "generator": x.generator, "source": x.source})
        mp = f"data/manifests/stress_{variant}.csv"
        with open(mp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "orig", "label", "generator", "source"])
            w.writeheader(); w.writerows(rows)
        print(f"{variant}: {len(rows)} rows -> {mp}", flush=True)


if __name__ == "__main__":
    main()

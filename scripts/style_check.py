"""Checkpoint-level style-reliance check (the canary_audit mitigation).

canon5 and canon6 both FAIL the manifest style canary (global tone/saturation/
grain/sharpness/vignette separate the classes at ~0.68 vs the 0.65 line). The
manifest gate cannot tell whether a *model* trained on that data actually reads
style, so the check moves to the checkpoint: score the same images with colour
destroyed (greyscale) and with the channels permuted (BGR). If AUROC survives
both, the decision is not a palette or colour-statistics shortcut.

    python -m scripts.style_check --manifest data/manifests/canon6_test.csv \
        --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt" --limit 1500
"""
from __future__ import annotations

import argparse
import random

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data import load_image, load_manifest
from src.model import load_model

BATCH = 32


def greyscale(im: Image.Image) -> Image.Image:
    return im.convert("L").convert("RGB")


def bgr(im: Image.Image) -> Image.Image:
    r, g, b = im.convert("RGB").split()
    return Image.merge("RGB", (b, g, r))


def rbg(im: Image.Image) -> Image.Image:
    r, g, b = im.convert("RGB").split()
    return Image.merge("RGB", (r, b, g))


VARIANTS = {"clean": lambda im: im, "greyscale": greyscale, "bgr_swap": bgr, "rbg_swap": rbg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    samples = load_manifest(a.manifest)
    if a.limit and len(samples) > a.limit:
        random.Random(a.seed).shuffle(samples)
        samples = samples[:a.limit]
    y = np.array([s.label for s in samples])
    model = load_model(a.model)
    print(f"{a.manifest}: n={len(samples)} ({int((y==0).sum())} real / {int((y==1).sum())} fake)")
    print(f"model: {a.model}\n")

    base = None
    for name, fn in VARIANTS.items():
        scores = []
        for i in range(0, len(samples), BATCH):
            imgs = []
            for s in samples[i:i + BATCH]:
                try:
                    imgs.append(fn(load_image(s.path)))
                except Exception:
                    imgs.append(Image.new("RGB", (176, 176)))
            scores.extend(model.predict(imgs))
        auc = roc_auc_score(y, np.asarray(scores))
        if base is None:
            base = auc
        print(f"  {name:10s} AUROC {auc:.4f}   delta vs clean {auc - base:+.4f}")

    print("\n  Reading: if greyscale and the channel swaps hold up, the decision is not "
          "colour/palette. A large drop under greyscale would mean colour statistics "
          "were doing the work -- i.e. the style canary's warning had teeth.")


if __name__ == "__main__":
    main()

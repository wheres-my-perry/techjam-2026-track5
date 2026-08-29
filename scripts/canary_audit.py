"""Canary audit: can a DELIBERATELY STUPID model classify this manifest?

    python -m scripts.canary_audit --manifest data/manifests/canon2_test.csv

Companion to scripts/shortcut_audit.py (which sees metadata only, never a
pixel). This one DOES see pixels -- but only through feature extractors so
weak that they cannot possibly represent "was this image AI-generated".
They can see average colour, coarse layout, overall contrast. Nothing else.

The logic: a canary has no access to generator fingerprints (high-frequency
texture, spectral grid, resampling traces). The ONLY way it can score well
is if reals and fakes differ in something dumb -- subject matter, colour
palette, brightness, framing. That is a dataset-design flaw, not detection.

    canary AUROC ~0.5     reals and fakes are drawn alike; benchmark is fair
    0.5-0.65              mild content skew: caveat results
    > 0.65                FAIL: the classes differ in CONTENT. A real model
                          will ride that skew and its score is not detection.

The size confound (2026-08-28) was this same failure with `size` as the dumb
variable. This audit generalises the check so the next one cannot hide.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageFilter

from src.data import load_image
from src.metrics import auroc


# --- canaries: each returns a fixed-length vector, all of them content-only ---

def f_color(im: Image.Image) -> list[float]:
    """8 dims: per-channel mean/std, brightness, saturation. Colour cast only."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    return [*a.mean((0, 1)), *a.std((0, 1)), float(a.mean()), float((mx - mn).mean())]


def f_hist(im: Image.Image) -> list[float]:
    """48 dims: 16-bin colour histogram per channel. Palette, no structure."""
    a = np.asarray(im, dtype=np.uint8)
    return [v for c in range(3)
            for v in np.histogram(a[:, :, c], bins=16, range=(0, 256), density=True)[0]]


def f_thumb8(im: Image.Image) -> list[float]:
    """64 dims: 8x8 greyscale thumbnail. Coarse layout; all texture destroyed."""
    t = im.convert("L").resize((8, 8), Image.BILINEAR)
    a = np.asarray(t, dtype=np.float32) / 255.0
    return list(a.ravel())


def f_blur(im: Image.Image) -> list[float]:
    """27 dims: heavy blur (sigma 8) then 3x3 grid of per-channel means.
    Generator fingerprints live in high frequencies; this keeps only content."""
    b = im.filter(ImageFilter.GaussianBlur(8))
    a = np.asarray(b, dtype=np.float32) / 255.0
    h, w, _ = a.shape
    ys = np.linspace(0, h, 4, dtype=int)
    xs = np.linspace(0, w, 4, dtype=int)
    return [float(a[ys[i]:ys[i + 1], xs[j]:xs[j + 1], c].mean())
            for i in range(3) for j in range(3) for c in range(3)]


def f_style(im: Image.Image) -> list[float]:
    """12 dims: global STYLE statistics only -- luminance percentiles/contrast, saturation
    mean/std, grain (high-pass energy), sharpness (Laplacian energy), vignette (centre vs
    corner luminance). No layout, no palette. If this separates classes, the model can
    read 'aesthetic' as the label (found 2026-08-29 in the reference benchmark)."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    lum = a.mean(2)
    p5, p50, p95 = np.percentile(lum, [5, 50, 95])
    sat = a.max(2) - a.min(2)
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32).mean(2) / 255.0
    grain = float(np.abs(lum - blur).mean())
    lap = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:])
    h, w = lum.shape
    c = lum[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    corners = np.mean([lum[:h // 4, :w // 4].mean(), lum[:h // 4, -w // 4:].mean(), lum[-h // 4:, :w // 4].mean(), lum[-h // 4:, -w // 4:].mean()])
    return [p5, p50, p95, p95 - p5, float(lum.std()), float(sat.mean()), float(sat.std()), grain,
            float(lap.mean()), float(lap.std()), float(c - corners), float((sat < 0.05).mean())]


CANARIES = {
    "style": (f_style, "global style stats: tone, saturation, grain, sharpness, vignette"),
    "color": (f_color, "mean/std colour + brightness + saturation"),
    "hist": (f_hist, "16-bin colour histogram per channel"),
    "thumb8": (f_thumb8, "8x8 greyscale thumbnail (layout only)"),
    "blur": (f_blur, "sigma-8 blur, 3x3 grid means (content only)"),
}


def verdict(a: float) -> str:
    return ("CLEAN" if a < 0.55 else
            "MILD CONTENT SKEW — caveat results" if a <= 0.65 else
            "FAIL — classes differ in content, not just in being fake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--canaries", default="all",
                    help="comma-separated subset of " + ",".join(CANARIES))
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the worst canary FAILs, so a Slurm "
                         "--dependency=afterok chain cannot start training on "
                         "a content-skewed manifest")
    ap.add_argument("--max-auroc", type=float, default=0.65)
    args = ap.parse_args()

    from src.data import load_manifest
    s = load_manifest(args.manifest)
    random.Random(0).shuffle(s)
    s = s[: args.limit]

    names = list(CANARIES) if args.canaries == "all" else args.canaries.split(",")
    feats = {n: [] for n in names}
    y, gens = [], []
    for x in s:
        try:
            im = load_image(os.path.expandvars(x.path))
        except Exception as e:
            print(f"skip {x.path}: {e}", flush=True)
            continue
        for n in names:
            feats[n].append(CANARIES[n][0](im))
        y.append(x.label)
        gens.append(x.generator or "real")
    y = np.asarray(y)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    print(f"\ncanary audit — {args.manifest}  ({len(y)} rows, "
          f"{int((y == 0).sum())} real / {int((y == 1).sum())} fake)\n")
    worst = 0.0
    for n in names:
        X = np.asarray(feats[n], dtype=np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        a = auroc(y, p)
        worst = max(worst, a)
        print(f"  {n:8s} ({X.shape[1]:3d}d)  AUROC {a:.4f}   {verdict(a)}")
        print(f"           {CANARIES[n][1]}")
        # which groups the canary separates -- points at the guilty subset
        rows = []
        for g in sorted(set(gens)):
            m = np.array([gg == g for gg in gens])
            if g == "real" or m.sum() < 20:
                continue
            mm = m | (np.array(gens) == "real")
            rows.append((auroc(y[mm], p[mm]), g, int(m.sum())))
        for a_g, g, k in sorted(rows, reverse=True):
            print(f"             vs real: {g:16s} {a_g:.3f}  (n={k})")
        print()

    print(f"WORST CANARY: {worst:.4f}  [{verdict(worst)}]", flush=True)
    if args.strict and worst > args.max_auroc:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

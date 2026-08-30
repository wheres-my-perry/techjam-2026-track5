"""Random-size cropping, shared by training AND inference.

Thinh's rule (2026-08-29): crop to a RANDOM SIZE, and use the same crop
procedure at train and at inference so there is no train/test mismatch.
Both paths import from here so the two cannot drift apart -- that mismatch
is a real bug we hit: training cropped 160 while the inference vote wrapper
cropped 224, which is LARGER than a canon2 image, so it upscaled 176->224
and reintroduced the resampling signature canonicalization exists to remove.

Two hard rules:
  * never crop larger than the image (no upscaling, ever);
  * crop at native resolution (no resampling), so nothing about scale
    correlates with the label.
"""

from __future__ import annotations

import random

CROP_MIN = 112
CROP_MAX = 176


def clamp_size(size: int, w: int, h: int) -> int:
    """Largest usable crop <= size that still fits inside the image."""
    return max(8, min(size, w, h))


def random_crop(img, size: int, rng: random.Random):
    """One random-position crop of `size` (clamped to fit). No resampling."""
    w, h = img.size
    c = clamp_size(size, w, h)
    x = rng.randint(0, w - c)
    y = rng.randint(0, h - c)
    return img.crop((x, y, x + c, y + c))


def snap(size: int, step: int) -> int:
    """Round DOWN to a multiple of `step` (ViT patch size; 1 = no constraint)."""
    return max(step, (size // step) * step) if step > 1 else size


def sample_size(rng: random.Random, cmin: int = CROP_MIN,
                cmax: int = CROP_MAX, step: int = 1) -> int:
    """Draw a crop size. Training draws one per BATCH (a batch has to stack
    into a single tensor), inference sweeps the same range. `step` snaps to a
    multiple (ViT-L/14 needs sides divisible by 14)."""
    return snap(rng.randint(min(cmin, cmax), max(cmin, cmax)), step)


def size_ladder(cmin: int = CROP_MIN, cmax: int = CROP_MAX, n: int = 3,
                step: int = 1):
    """Deterministic sizes spanning the training range, for inference.

    Training sees sizes ~U[cmin, cmax]; inference covers that same range on a
    fixed ladder so scores are reproducible instead of randomly varying.
    """
    if n <= 1 or cmax <= cmin:
        return [snap(cmax, step)]
    inc = (cmax - cmin) / (n - 1)
    return sorted({snap(int(round(cmin + i * inc)), step) for i in range(n)})


def grid_boxes(w: int, h: int, size: int, grid: int = 3, step: int = 1):
    """Boxes (x0, y0, x1, y1) of all `grid`x`grid` evenly spaced crops of
    `size` (clamped, no upscale). Same order as grid_views."""
    c = snap(clamp_size(size, w, h), step)
    xs = sorted({round(t * (w - c) / max(1, grid - 1)) for t in range(grid)})
    ys = sorted({round(t * (h - c) / max(1, grid - 1)) for t in range(grid)})
    return [(x, y, x + c, y + c) for y in ys for x in xs]


def tile_boxes(w: int, h: int, size: int, m: int = 1, step: int = 1):
    """Even-coverage tiling (Thinh, 2026-08-30): m*m shifted partitions of the image into
    non-overlapping `size` squares. Partition (i, j) starts at (i*size/m, j*size/m); the last
    tile on each axis is clamped to the far edge so nothing is left uncovered. Interior pixels
    are covered exactly m*m times; deterministic. m=1 is a plain partition."""
    c = snap(clamp_size(size, w, h), step)

    def axis(L):
        pos = set()
        for o in range(m):
            x = round(o * c / m)
            while x + c <= L:
                pos.add(x); x += c
        pos.add(L - c)
        return sorted(pos)
    return [(x, y, x + c, y + c) for y in axis(h) for x in axis(w)]


def grid_views(img, size: int, grid: int = 3, step: int = 1):
    """All `grid`x`grid` evenly spaced crops of `size` (clamped, no upscale)."""
    return [img.crop(b) for b in grid_boxes(*img.size, size, grid, step)]

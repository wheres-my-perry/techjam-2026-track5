"""Contest transform grid (Track 5) + random train-time augmentation.

All transforms take and return a PIL RGB Image. PIL-only (+numpy) — no cv2.
Eval grid parameters come verbatim from the official brief.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------- transforms

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    # PIL's GaussianBlur radius parameter is the standard deviation.
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """sigma is in [0,1] scale (brief: 0.02/0.05/0.10)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = np.random.default_rng(0).normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


def color_jitter(img: Image.Image, factor: float) -> Image.Image:
    """Deterministic worst-case-ish jitter: brightness/contrast/saturation all shifted by +factor."""
    img = ImageEnhance.Brightness(img).enhance(1.0 + factor)
    img = ImageEnhance.Contrast(img).enhance(1.0 + factor)
    img = ImageEnhance.Color(img).enhance(1.0 + factor)
    return img


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


# ---------------------------------------------------------------- eval grid

def _named(name, fn):
    fn.__grid_name__ = name
    return name, fn


EVAL_GRID = [
    _named("clean", lambda im: im),
    _named("jpeg_q90", lambda im: jpeg_compress(im, 90)),
    _named("jpeg_q70", lambda im: jpeg_compress(im, 70)),
    _named("jpeg_q50", lambda im: jpeg_compress(im, 50)),
    _named("jpeg_q30", lambda im: jpeg_compress(im, 30)),
    _named("blur_s0.5", lambda im: gaussian_blur(im, 0.5)),
    _named("blur_s1.0", lambda im: gaussian_blur(im, 1.0)),
    _named("blur_s2.0", lambda im: gaussian_blur(im, 2.0)),
    _named("resize_0.5x", lambda im: resize_down_up(im, 0.5)),
    _named("resize_0.25x", lambda im: resize_down_up(im, 0.25)),
    _named("noise_s0.02", lambda im: gaussian_noise(im, 0.02)),
    _named("noise_s0.05", lambda im: gaussian_noise(im, 0.05)),
    _named("noise_s0.10", lambda im: gaussian_noise(im, 0.10)),
    _named("jitter_20", lambda im: color_jitter(im, 0.20)),
    _named("crop_80", lambda im: center_crop(im, 0.80)),
]


# ------------------------------------------------- random train-time version

def random_train_transform(img: Image.Image, rng: random.Random,
                           geometry: bool = True) -> Image.Image:
    """Apply 0-2 random transforms with parameters sampled from the eval ranges.

    Mirrors the eval distribution so the model trains on what it is tested on.
    """
    ops = []
    if rng.random() < 0.5:
        ops.append(lambda im: jpeg_compress(im, rng.randint(30, 90)))
    if rng.random() < 0.3:
        ops.append(lambda im: gaussian_blur(im, rng.uniform(0.0, 2.0)))
    if rng.random() < 0.3:
        ops.append(lambda im: resize_down_up(im, rng.uniform(0.25, 0.75)))
    if rng.random() < 0.3:
        arr = None  # noise applied inline below to use rng
        sigma = rng.uniform(0.01, 0.10)
        def _noise(im, s=sigma, r=rng):
            a = np.asarray(im, dtype=np.float32) / 255.0
            n = np.random.default_rng(r.randrange(2**32)).normal(0, s, a.shape).astype(np.float32)
            return Image.fromarray((np.clip(a + n, 0, 1) * 255 + 0.5).astype(np.uint8))
        ops.append(_noise)
    if rng.random() < 0.3:
        f = rng.uniform(-0.2, 0.2)
        ops.append(lambda im: color_jitter(im, f))
    if geometry and rng.random() < 0.3:   # geometry=False keeps size (mask-aligned training)
        frac = rng.uniform(0.8, 1.0)
        ops.append(lambda im: center_crop(im, frac))
    rng.shuffle(ops)
    for op in ops[:2]:  # at most 2 stacked, like real repost chains
        img = op(img)
    return img

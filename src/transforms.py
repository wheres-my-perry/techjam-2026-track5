"""Contest transform grid (Track 5) + random train-time augmentation.

All transforms take and return a PIL RGB Image. PIL-only (+numpy) — no cv2.
The evaluation grid follows the brief's listed families and scalar settings except
for color jitter: `jitter_20` deterministically raises brightness, contrast, and
saturation together by 20%, so it does not cover the brief's full ±20% range.
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

# Stacked conditions (2026-08-30, Thinh: the brief says "a subset of the following augmentations" —
# that limits WHICH transforms, not how many per image; repost chains are stacks). Reported alongside
# the single-transform grid; not inside EVAL_GRID so earlier single-transform numbers stay comparable.
def _seeded(im):
    import hashlib
    return random.Random(int(hashlib.md5(im.tobytes()[:4096]).hexdigest(), 16) & 0xFFFFFFFF)


def _stack(im, k, rng=None):
    """k transforms drawn from the brief's grid. Evaluation: seeded by image content (deterministic).
    Training (--stack-aug): pass a fresh random.Random() so every epoch draws a different stack."""
    rng = rng if rng is not None else _seeded(im)
    pool = [lambda i: jpeg_compress(i, rng.choice([90, 70, 50, 30])),
            lambda i: gaussian_blur(i, rng.choice([0.5, 1.0, 2.0])),
            lambda i: resize_down_up(i, rng.choice([0.5, 0.25])),
            lambda i: gaussian_noise(i, rng.choice([0.02, 0.05, 0.10])),
            lambda i: color_jitter(i, 0.20),
            lambda i: center_crop(i, 0.80)]
    for op in rng.sample(pool, k):
        im = op(im)
    return im


def stack_no_geometry(im, k, rng=None):
    """k DISTINCT transforms that all preserve image size, for consistency-training views.

    The K views of one crop are stacked into a single tensor, so they must come out the same size;
    that rules out centre-crop and caps the depth at the five size-preserving families. Added
    2026-09-01 after --stack-aug was found to have no effect in consistency mode.
    """
    rng = rng if rng is not None else _seeded(im)
    pool = [lambda i: jpeg_compress(i, rng.choice([90, 70, 50, 30])),
            lambda i: gaussian_blur(i, rng.choice([0.5, 1.0, 2.0])),
            lambda i: resize_down_up(i, rng.choice([0.5, 0.25])),
            lambda i: gaussian_noise(i, rng.choice([0.02, 0.05, 0.10])),
            lambda i: color_jitter(i, 0.20)]
    for op in rng.sample(pool, min(k, len(pool))):
        im = op(im)
    return im


EXTRA_GRID = [
    _named("chain_repost", lambda im: jpeg_compress(resize_down_up(jpeg_compress(im, 70), 0.5), 50)),
    _named("jpeg_twice", lambda im: jpeg_compress(jpeg_compress(im, 50), 50)),
    _named("blur1_jpeg70", lambda im: jpeg_compress(gaussian_blur(im, 1.0), 70)),
    _named("noise05_jpeg70", lambda im: jpeg_compress(gaussian_noise(im, 0.05), 70)),
    _named("crop80_resize05", lambda im: resize_down_up(center_crop(im, 0.80), 0.5)),
    # depth 1 completes the ladder clean -> 1 -> ... -> 6 for scripts.depth_ladder
    _named("stack1_rand", lambda im: _stack(im, 1)),
    _named("stack2_rand", lambda im: _stack(im, 2)),
    _named("stack3_rand", lambda im: _stack(im, 3)),
    # 2026-08-31 (Thinh): "a subset of the following augmentations" limits WHICH of the six
    # transform families may be used, not how many are composed on one image -- so a subset can
    # be any size up to all six. Depths 4-6 report the far end of that reading.
    _named("stack4_rand", lambda im: _stack(im, 4)),
    _named("stack5_rand", lambda im: _stack(im, 5)),
    _named("stack6_rand", lambda im: _stack(im, 6)),
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


def hard_train_transform(img: Image.Image, rng: random.Random) -> Image.Image:
    """ONE extreme corruption from the hard end of the eval grid (option B2, 2026-08-30).

    Applied to BOTH classes with the same probability (class-neutral, so it cannot encode the
    label). Motivation: at a fixed cut-off the canon4 model flags 20-27% of real photos under
    resize 1/4, noise 0.10 or blur s2 -- it rarely saw reals that degraded during training.
    """
    k = rng.randrange(4)
    if k == 0:
        return gaussian_blur(img, rng.uniform(1.5, 2.5))
    if k == 1:
        return resize_down_up(img, rng.uniform(0.2, 0.4))
    if k == 2:
        a = np.asarray(img, dtype=np.float32) / 255.0
        n = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.07, 0.12), a.shape).astype(np.float32)
        return Image.fromarray((np.clip(a + n, 0, 1) * 255 + 0.5).astype(np.uint8))
    return jpeg_compress(img, rng.randint(20, 40))


# ------------------------------------------------- style neutralisation (2026-08-29)

def style_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    """Label-NEUTRAL style randomisation, applied identically to reals and fakes.

    Error analysis on the reference benchmark showed the model reading aesthetic as
    evidence: polished / HDR / B&W / flat real photos scored "AI", DALL-E images styled
    as flash party snapshots scored "real". Style correlates with the label in every
    dataset but is not caused by generation. Randomising it for BOTH classes removes
    the information, the same cure that worked for size and content shortcuts.
    0-2 of: greyscale, saturation, contrast/gamma, film grain, vignette, flash falloff.
    """
    from PIL import ImageEnhance
    ops = []
    if rng.random() < 0.15:
        ops.append(lambda im: im.convert("L").convert("RGB"))
    if rng.random() < 0.35:
        f = rng.uniform(0.4, 1.8)
        ops.append(lambda im, f=f: ImageEnhance.Color(im).enhance(f))
    if rng.random() < 0.35:
        c, g = rng.uniform(0.7, 1.5), rng.uniform(0.7, 1.4)
        def _tone(im, c=c, g=g):
            im = ImageEnhance.Contrast(im).enhance(c)
            a = np.asarray(im, dtype=np.float32) / 255.0
            return Image.fromarray((np.clip(a, 0, 1) ** g * 255 + 0.5).astype(np.uint8))
        ops.append(_tone)
    if rng.random() < 0.30:
        sig = rng.uniform(0.02, 0.08)
        def _grain(im, s=sig, r=rng):
            a = np.asarray(im, dtype=np.float32) / 255.0
            g = np.random.default_rng(r.randrange(2**32)).normal(0, s, a.shape[:2]).astype(np.float32)
            return Image.fromarray((np.clip(a + g[..., None], 0, 1) * 255 + 0.5).astype(np.uint8))
        ops.append(_grain)
    if rng.random() < 0.25:
        k = rng.uniform(0.3, 0.8)
        def _vignette(im, k=k):
            a = np.asarray(im, dtype=np.float32) / 255.0
            h, w = a.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
            return Image.fromarray((np.clip(a * (1 - k * d ** 2)[..., None], 0, 1) * 255 + 0.5).astype(np.uint8))
        ops.append(_vignette)
    if rng.random() < 0.15:
        def _flash(im, r=rng):
            a = np.asarray(im, dtype=np.float32) / 255.0
            h, w = a.shape[:2]
            cy, cx = r.uniform(0.3, 0.7) * h, r.uniform(0.3, 0.7) * w
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
            gain = 1.4 * np.exp(-(d / 0.35) ** 2) + 0.35
            return Image.fromarray((np.clip(a * gain[..., None], 0, 1) * 255 + 0.5).astype(np.uint8))
        ops.append(_flash)
    rng.shuffle(ops)
    for op in ops[:2]:
        img = op(img)
    return img

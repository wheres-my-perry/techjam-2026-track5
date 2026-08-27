"""Spectral detector (approach 03): FFT artifact features + logistic head.

Rationale: generator decoders (transposed conv, VQ token grids, upsamplers)
leave PERIODIC energy in the frequency domain — checkerboard peaks on the
axes, token-grid harmonics, azimuthal spikes — that natural camera images
lack. Kill-test target: the vqvae/TOKEN family (0.53-0.68 for every learned
model so far). Prediction (GENERATOR_MATRIX): GAN strong, TOKEN mid-strong,
DIFF weak-mid.

Two views per image, 12 features each (24-dim total):
- center-crop 256 at NATIVE resolution — pixel-level periodicity survives
  cropping but is destroyed by resizing, so this is the artifact view;
- whole image resized to 256 — global spectral shape, resolution-normalized.
Features per view: 8 radial band-energy fractions of the high-pass residual
spectrum, 3 axis peak-to-background ratios (Nyquist, N/4, N/8 checkerboard
lines), 1 azimuthal non-uniformity (CoV over 16 angular sectors, high band).

No torch, no GPU: numpy + PIL + scikit-learn (train only). Fast on CPU.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from src.model import BaseModel

N = 256  # analysis size for both views
_BAND_EDGES = [0, 8, 16, 32, 48, 64, 96, 128]
_PEAK_FREQS = [128, 64, 32]  # Nyquist line, N/4, N/8 (checkerboard harmonics)


def _center_crop_native(img: Image.Image) -> Image.Image:
    w, h = img.size
    if min(w, h) < N:  # tiny image: minimal upscale so a 256 crop exists
        s = N / min(w, h)
        img = img.resize((max(N, round(w * s)), max(N, round(h * s))))
        w, h = img.size
    x, y = (w - N) // 2, (h - N) // 2
    return img.crop((x, y, x + N, y + N))


def _view_features(lum: np.ndarray) -> list[float]:
    """12 spectral features from one 256x256 luminance array in [0,1]."""
    im = Image.fromarray((lum * 255).astype(np.uint8))
    blurred = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)),
                         dtype=np.float32) / 255.0
    r = lum - blurred  # high-pass residual: artifacts live here
    P = np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2
    c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(yy - c, xx - c)
    total = P.sum() + 1e-12

    feats: list[float] = []
    for lo, hi in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:]):
        feats.append(float(P[(rr >= lo) & (rr < hi)].sum() / total))

    def _win(y, x):  # mean power in 3x3 window, clipped to array
        y0, y1 = max(0, y - 1), min(N, y + 2)
        x0, x1 = max(0, x - 1), min(N, x + 2)
        return float(P[y0:y1, x0:x1].mean())

    for f in _PEAK_FREQS:
        peaks = [_win(c, min(N - 1, c + f)), _win(c, max(0, c - f)),
                 _win(min(N - 1, c + f), c), _win(max(0, c - f), c)]
        ann = P[(rr >= f - 3) & (rr <= f + 3)]
        background = float(np.median(ann)) + 1e-12
        feats.append(float(np.log1p(max(peaks) / background)))

    hi_band = (rr >= 96) & (rr < 128)
    theta = np.arctan2(yy - c, xx - c)[hi_band]
    energy = P[hi_band]
    sectors = np.floor((theta + np.pi) / (2 * np.pi) * 16).astype(int) % 16
    sums = np.bincount(sectors, weights=energy, minlength=16)
    feats.append(float(sums.std() / (sums.mean() + 1e-12)))
    return feats


def features(img: Image.Image) -> np.ndarray:
    """24-dim spectral feature vector. Deterministic."""
    img = img.convert("RGB")
    feats: list[float] = []
    for view in (_center_crop_native(img), img.resize((N, N), Image.BILINEAR)):
        lum = np.asarray(view, dtype=np.float32).mean(axis=2) / 255.0
        feats += _view_features(lum)
    return np.asarray(feats, dtype=np.float64)


class SpectralModel(BaseModel):
    name = "spectral"

    def __init__(self, weights_path: str = "outputs/spectral/baseline.npz"):
        z = np.load(weights_path)
        self.w = z["w"]
        self.b = float(z["b"])
        self.f_mean = z["f_mean"]
        self.f_std = z["f_std"]

    def predict(self, images):
        scores = []
        for im in images:
            z = (features(im) - self.f_mean) / self.f_std
            logit = float(z @ self.w) + self.b
            scores.append(1.0 / (1.0 + np.exp(-logit)))
        return np.asarray(scores, dtype=np.float32)

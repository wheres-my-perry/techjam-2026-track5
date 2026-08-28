"""Real-manifold anomaly detector: trained on REALS ONLY.

Fingerprint = camera-pipeline signal statistics (noise residuals across scales,
noise-field uniformity, radial spectrum). One-class fit = robust Gaussian
(Ledoit-Wolf covariance) over standardized features; score = Mahalanobis
distance mapped to [0,1] by the rank of that distance among training reals.

No torch, no GPU: numpy + PIL + scikit-learn. Runs anywhere, fast on CPU.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import kurtosis

from src.model import BaseModel

SPECTRAL_SIZE = 256  # fixed size for FFT features only (stats features use native res)


def _residual_stats(arr: np.ndarray, blurred: np.ndarray) -> list[float]:
    r = arr - blurred
    out = []
    for c in range(3):
        ch = r[..., c].ravel()
        out += [float(ch.std()), float(np.abs(ch).mean()), float(kurtosis(ch))]
    return out


def features(img: Image.Image) -> np.ndarray:
    """~32-dim camera-fingerprint feature vector. Deterministic."""
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    feats: list[float] = []
    blurs = {}
    for sigma in (1.0, 2.0):
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma)),
                       dtype=np.float32) / 255.0
        blurs[sigma] = b
        feats += _residual_stats(arr, b)              # fine + medium residuals
    feats += _residual_stats(blurs[1.0], blurs[2.0])  # band between scales

    # noise-field uniformity: CoV of per-patch residual stds (4x4 grid, luminance)
    lum_r = (arr - blurs[1.0]).mean(axis=2)
    H, W = lum_r.shape
    ph, pw = max(1, H // 4), max(1, W // 4)
    stds = [lum_r[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].std()
            for i in range(4) for j in range(4)]
    stds = np.asarray(stds)
    feats.append(float(stds.std() / (stds.mean() + 1e-8)))

    # radial spectrum: energy fractions in 4 bands (on fixed-size luminance)
    small = np.asarray(img.resize((SPECTRAL_SIZE, SPECTRAL_SIZE), Image.BILINEAR),
                       dtype=np.float32).mean(axis=2) / 255.0
    f = np.abs(np.fft.fftshift(np.fft.fft2(small - small.mean()))) ** 2
    yy, xx = np.mgrid[0:SPECTRAL_SIZE, 0:SPECTRAL_SIZE]
    rr = np.hypot(yy - SPECTRAL_SIZE / 2, xx - SPECTRAL_SIZE / 2)
    total = f.sum() + 1e-8
    edges = [0, 16, 48, 96, SPECTRAL_SIZE]
    for lo, hi in zip(edges[:-1], edges[1:]):
        feats.append(float(f[(rr >= lo) & (rr < hi)].sum() / total))

    # flat/saturated crops make kurtosis NaN (constant residual); neutralize so
    # crop-level use (voting, stacking) never crashes downstream sklearn.
    return np.nan_to_num(np.asarray(feats, dtype=np.float64),
                         nan=0.0, posinf=1e6, neginf=-1e6)


class RealManifoldModel(BaseModel):
    name = "real_manifold"

    def __init__(self, weights_path: str = "outputs/real_manifold/baseline.npz"):
        z = np.load(weights_path)
        self.mu = z["mu"]
        self.prec = z["prec"]           # precision matrix (inv covariance)
        self.f_mean = z["f_mean"]
        self.f_std = z["f_std"]
        self.train_d = z["train_d"]     # sorted train-real distances (rank map)

    def _distance(self, x: np.ndarray) -> float:
        z = (x - self.f_mean) / self.f_std
        d = z - self.mu
        return float(d @ self.prec @ d)

    def predict(self, images):
        # score = d / (d + median_train_distance): monotone in distance, in (0,1),
        # ~0.5 at the typical real, never saturates (no tied scores for far outliers).
        d_med = float(np.median(self.train_d)) + 1e-8
        scores = []
        for im in images:
            d = self._distance(features(im))
            scores.append(d / (d + d_med))
        return np.asarray(scores, dtype=np.float32)


def fit(feature_matrix: np.ndarray):
    """Fit the one-class model on real-image features. Returns dict of arrays."""
    from sklearn.covariance import LedoitWolf
    f_mean = feature_matrix.mean(axis=0)
    f_std = feature_matrix.std(axis=0) + 1e-8
    Z = (feature_matrix - f_mean) / f_std
    lw = LedoitWolf().fit(Z)
    mu = lw.location_
    prec = lw.get_precision()
    diffs = Z - mu
    train_d = np.sort(np.einsum("ij,jk,ik->i", diffs, prec, diffs))
    return {"mu": mu, "prec": prec, "f_mean": f_mean, "f_std": f_std,
            "train_d": train_d}


def save(params: dict, path: str, meta: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **params)
    if meta is not None:
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=1)

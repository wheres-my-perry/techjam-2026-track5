"""Model interface. Every model exposes: predict(images) -> scores in [0,1].

score = P(image is AI-generated). Swap implementations behind load_model().
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class BaseModel:
    name = "base"

    def predict(self, images: list[Image.Image]) -> np.ndarray:  # (N,) floats in [0,1]
        raise NotImplementedError


class RandomModel(BaseModel):
    """Deterministic pseudo-random scores (hash of image bytes). AUROC ~= 0.5.

    Exists so the whole harness runs end-to-end before any real model exists.
    """

    name = "random"

    def predict(self, images):
        scores = []
        for im in images:
            h = hashlib.md5(im.resize((32, 32)).tobytes()).digest()
            scores.append(int.from_bytes(h[:4], "big") / 2**32)
        return np.asarray(scores, dtype=np.float32)


class MeanBrightnessModel(BaseModel):
    """Trivial non-random baseline (scores by brightness). Only for harness testing."""

    name = "brightness"

    def predict(self, images):
        return np.asarray(
            [np.asarray(im, dtype=np.float32).mean() / 255.0 for im in images],
            dtype=np.float32,
        )


_REGISTRY = {
    "random": RandomModel,
    "brightness": MeanBrightnessModel,
    # "clip_linear": ...  (candidate v1 real model)
}


# Approaches register here: name -> (module path, class name, default weights).
# Modules are imported lazily so heavy deps (torch) load only when that approach is used.
_APPROACHES = {
    "cnn": ("src.approaches.cnn.model", "CNNModel", "outputs/cnn/baseline.pt"),
    "clip_linear": ("src.approaches.clip_linear.model", "CLIPLinearModel", "outputs/clip_linear/baseline.pt"),
    "resnet_ft": ("src.approaches.resnet_ft.model", "ResNetFTModel", "outputs/resnet_ft/baseline.pt"),
    "real_manifold": ("src.approaches.real_manifold.model", "RealManifoldModel", "outputs/real_manifold/baseline.npz"),
    "spectral": ("src.approaches.spectral.model", "SpectralModel", "outputs/spectral/baseline.npz"),
}


class CropVoteModel(BaseModel):
    """Inference-time patch voting (approach 01, stage 1 — no training needed).

    Scores a grid of fixed-size crops at native resolution through any inner
    model and aggregates by top-k mean: "an image is fake if some regions are
    fake." Fixes the pooling-dilution failure measured on full-resolution eval.
    All crops share one size -> inner model batches them efficiently.
    """

    def __init__(self, inner: BaseModel, crop=224, grid=3, topk=3):
        self.inner = inner
        self.name = f"vote+{inner.name}"
        self.crop, self.grid, self.topk = crop, grid, topk

    def _views(self, im):
        c = self.crop
        w, h = im.size
        if min(w, h) < c:  # tiny image: upscale short side, single center view
            s = c / min(w, h)
            im = im.resize((max(c, round(w * s)), max(c, round(h * s))))
            w, h = im.size
        xs = sorted({round(t * (w - c) / max(1, self.grid - 1)) for t in range(self.grid)})
        ys = sorted({round(t * (h - c) / max(1, self.grid - 1)) for t in range(self.grid)})
        return [im.crop((x, y, x + c, y + c)) for y in ys for x in xs]

    def predict(self, images):
        views, owners = [], []
        for i, im in enumerate(images):
            vs = self._views(im)
            views.extend(vs)
            owners.extend([i] * len(vs))
        vscores = self.inner.predict(views)
        out = np.zeros(len(images), dtype=np.float32)
        owners = np.asarray(owners)
        for i in range(len(images)):
            s = np.sort(vscores[owners == i])[::-1]
            k = min(self.topk, len(s))
            out[i] = float(s[:k].mean())
        return out


def load_model(name: str = "random") -> BaseModel:
    """Load by name. 'name:path.pt' picks weights; 'vote+name:path.pt' wraps any
    model in inference-time crop voting (grid crops, top-k aggregation)."""
    if name.startswith("vote+"):
        return CropVoteModel(load_model(name[len("vote+"):]))
    base, _, path = name.partition(":")
    if base in _APPROACHES:
        import importlib
        mod_path, cls_name, default_weights = _APPROACHES[base]
        cls = getattr(importlib.import_module(mod_path), cls_name)
        return cls(path or default_weights)
    if base in _REGISTRY:
        return _REGISTRY[base]()
    known = sorted(_REGISTRY) + [f"{k}[:weights]" for k in _APPROACHES]
    raise ValueError(f"Unknown model '{name}'. Known: {known}")

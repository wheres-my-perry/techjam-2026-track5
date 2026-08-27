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
}


def load_model(name: str = "random") -> BaseModel:
    """Load by name. Approach models accept 'name:path/to/weights.pt' to pick weights."""
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

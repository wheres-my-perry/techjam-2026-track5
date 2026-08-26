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
    # "clip_linear": ...  (v1 real model goes here)
}


def load_model(name: str = "random") -> BaseModel:
    if name in _REGISTRY:
        return _REGISTRY[name]()
    raise ValueError(f"Unknown model '{name}'. Known: {sorted(_REGISTRY)}")

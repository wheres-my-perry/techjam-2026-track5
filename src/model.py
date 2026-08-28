"""Model interface. Every model exposes: predict(images) -> scores in [0,1].

score = P(image is AI-generated). Swap implementations behind load_model().
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from src.crops import CROP_MAX, CROP_MIN, grid_views, size_ladder


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
    "pe_ft": ("src.approaches.pe_ft.model", "PEFTModel", "outputs/pe_ft/baseline.pt"),
    "real_manifold": ("src.approaches.real_manifold.model", "RealManifoldModel", "outputs/real_manifold/baseline.npz"),
    "spectral": ("src.approaches.spectral.model", "SpectralModel", "outputs/spectral/baseline.npz"),
    "patch_relation": ("src.approaches.patch_relation.model", "PatchRelationModel", "outputs/patch_relation/baseline.pt"),
    "stacked": ("src.approaches.stacked.model", "StackedModel", "outputs/stacked/baseline.npz"),
}


class CropVoteModel(BaseModel):
    """Inference-time patch voting (approach 01, stage 1 — no training needed).

    Scores a grid of fixed-size crops at native resolution through any inner
    model and aggregates by top-k mean: "an image is fake if some regions are
    fake." Fixes the pooling-dilution failure measured on full-resolution eval.
    All crops share one size -> inner model batches them efficiently.
    """

    def __init__(self, inner: BaseModel, cmin=None, cmax=None,
                 grid=3, topk=3, n_sizes=3):
        self.inner = inner
        self.name = f"vote+{inner.name}"
        # an approach may declare its own crop range/step (a ViT-L/14 needs
        # sides divisible by 14); default to the shared range otherwise
        self.cmin = cmin if cmin is not None else getattr(inner, "CROP_MIN", CROP_MIN)
        self.cmax = cmax if cmax is not None else getattr(inner, "CROP_MAX", CROP_MAX)
        self.step = getattr(inner, "CROP_STEP", 1)
        self.grid, self.topk, self.n_sizes = grid, topk, n_sizes

    def _views(self, im):
        """Grid crops at several sizes spanning the TRAINING crop range.

        Training draws a random size per batch from [cmin, cmax]; inference
        sweeps the same range on a fixed ladder, so the two match while
        scores stay reproducible. src.crops never upscales -- the old code
        used a flat crop=224 and upscaled any smaller image to reach it,
        which put the resampling signature back into canon2's 176px inputs.
        """
        # Inputs smaller than the training range (the grid's resize_0.25x turns
        # a 176px image into 44px) are upscaled to CROP_MIN. This is the ONE
        # sanctioned upscale: it applies to every tiny input regardless of label,
        # so it cannot encode the label -- unlike upscaling a dataset whose
        # classes differ in native size. Below CROP_MIN the model is simply
        # out of range; measured 2026-08-29: resize_0.25x was the worst cell
        # (0.647 official) for exactly this reason.
        w, h = im.size
        if min(w, h) < self.cmin:
            sc = self.cmin / min(w, h)
            im = im.resize((max(self.cmin, round(w * sc)), max(self.cmin, round(h * sc))),
                           Image.BICUBIC)
        views = []
        for c in size_ladder(self.cmin, self.cmax, self.n_sizes, self.step):
            views += grid_views(im, c, self.grid, self.step)
        return views

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


class StandardizeWrapModel(BaseModel):
    """Anti-shortcut wrapper: resize EVERY input's short side to a fixed value
    (both classes, up- and down-scaling alike) so image size — and the
    tiny-image upscale path — can never act as a class cue. Made necessary by
    the 2026-08-28 finding that official-benchmark reals were all 200x200
    thumbnails vs 1024+ fakes (size alone separated the classes)."""

    def __init__(self, inner: BaseModel, short=512):
        self.inner = inner
        self.short = short
        self.name = f"std+{inner.name}"

    def predict(self, images):
        outs = []
        for im in images:
            w, h = im.size
            sc = self.short / min(w, h)
            outs.append(im.resize((max(1, round(w * sc)), max(1, round(h * sc))),
                                  Image.LANCZOS))
        return self.inner.predict(outs)


class NoiseWrapModel(BaseModel):
    """Observation #12 kill-test: the eval grid's heaviest-noise rows scored a
    paradoxical AUROC ~1.0, so try noise as a deliberate inference-time
    canonicalizer: add fixed Gaussian noise to every input before scoring.
    Deterministic (fixed RandomState) so evals are reproducible."""

    def __init__(self, inner: BaseModel, sigma=0.10):
        self.inner = inner
        self.sigma = sigma
        self.name = f"noise+{inner.name}"

    def predict(self, images):
        rng = np.random.RandomState(0)
        noisy = []
        for im in images:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            a = np.clip(a + rng.normal(0.0, self.sigma, a.shape), 0.0, 1.0)
            noisy.append(Image.fromarray((a * 255).astype(np.uint8)))
        return self.inner.predict(noisy)


def load_model(name: str = "random") -> BaseModel:
    """Load by name. 'name:path.pt' picks weights; 'vote+name:path.pt' wraps any
    model in inference-time crop voting (grid crops, top-k aggregation)."""
    if name.startswith("vote+"):
        return CropVoteModel(load_model(name[len("vote+"):]))
    if name.startswith("noise+"):
        return NoiseWrapModel(load_model(name[len("noise+"):]))
    if name.startswith("std+"):
        return StandardizeWrapModel(load_model(name[len("std+"):]))
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

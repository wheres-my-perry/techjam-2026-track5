"""Sanity tests for the eval harness. Run: python -m pytest tests/ -q"""

import random

import numpy as np
from PIL import Image

from src import metrics
from src.transforms import EVAL_GRID, random_train_transform


def _img(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_grid_has_15_conditions_including_clean():
    # brief grid: jpeg x4, blur x3, resize x2, noise x3, jitter x1, crop x1 = 14 (+ clean)
    names = [n for n, _ in EVAL_GRID]
    assert len(names) == 15
    assert names[0] == "clean"
    assert len(set(names)) == len(names)


def test_transforms_return_rgb_images():
    im = _img()
    for name, tf in EVAL_GRID:
        out = tf(im)
        assert out.mode == "RGB", name
        if name != "crop_80":
            assert out.size == im.size, name
        else:
            assert out.size == (int(64 * 0.8), int(48 * 0.8))


def test_transforms_actually_change_pixels():
    im = _img()
    for name, tf in EVAL_GRID:
        if name == "clean":
            continue
        assert tf(im).tobytes() != im.tobytes(), f"{name} was a no-op"


def test_random_train_transform_runs():
    rng = random.Random(0)
    for _ in range(20):
        out = random_train_transform(_img(), rng)
        assert out.mode == "RGB"


def test_auroc_perfect_and_random():
    y = [0, 0, 0, 1, 1, 1]
    assert metrics.auroc(y, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0
    assert metrics.auroc(y, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 0.0
    assert abs(metrics.auroc([0, 1] * 50, [0.5] * 100) - 0.5) < 1e-9


def test_auroc_matches_pairwise_definition():
    # worked example from our discussion: fakes {0.9,0.8,0.4}, reals {0.7,0.3,0.1} -> 8/9
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.4, 0.7, 0.3, 0.1]
    assert abs(metrics.auroc(y, s) - 8 / 9) < 1e-9


def test_balanced_accuracy_and_fpr():
    y = [0, 0, 1, 1]
    s = [0.1, 0.6, 0.7, 0.9]
    assert metrics.balanced_accuracy(y, s, 0.65) == 1.0
    # TPR hits 1.0 at thr=0.7; at that threshold no real (0.1, 0.6) is flagged -> FPR 0
    assert metrics.fpr_at_tpr(y, s, 0.95) == 0.0
    # but when catching all fakes forces thr down to 0.6, the real at 0.7 gets flagged
    assert metrics.fpr_at_tpr([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.95) == 0.5


def test_pick_threshold_separates_perfectly_separable():
    y = [0, 0, 1, 1]
    s = [0.2, 0.3, 0.8, 0.9]
    thr = metrics.pick_threshold(y, s)
    assert metrics.balanced_accuracy(y, s, thr) == 1.0

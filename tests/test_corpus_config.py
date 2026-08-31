"""Automated checks that the BUILT manifests satisfy configs/canon6.yaml.

Thinh's rule (2026-08-31): everything involved in training lives in one config, and is then
tested automatically. Every invariant here corresponds to a bug this project actually shipped:

  * holdout leakage        ddpm re-entered canon2 train through ArtiFact's own ddpm folder
  * partial edits in train  a whole-image label is wrong for a localized edit
  * forbidden slices        184 COCO val2017 rows (the judges' real class) sat inside ArtiFact
  * bucket imbalance        size predicted the label; the shrink factor is a physical trace
  * one-sided content       'bedroom = fake' at 92.7:1, then 12.55:1 from the opposite fix
  * cross-split files       the same source file in train and test

Run:  python -m pytest tests/test_corpus_config.py -q
Skips cleanly when the manifests are not present (e.g. on a laptop with no data).
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

import pytest
import yaml

CONFIG = os.path.join("configs", "canon6.yaml")
PREFIX = os.path.join("data", "manifests", "canon6")
SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def cfg():
    if not os.path.exists(CONFIG):
        pytest.skip("configs/canon6.yaml not present")
    return yaml.safe_load(open(CONFIG))


@pytest.fixture(scope="module")
def man():
    out = {}
    for sp in SPLITS:
        p = f"{PREFIX}_{sp}.csv"
        if not os.path.exists(p):
            pytest.skip(f"{p} not built")
        out[sp] = list(csv.DictReader(open(p, newline="")))
    return out


def bucket(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def test_holdout_generators_absent_from_train_and_val(cfg, man):
    held = set(cfg["routing"]["holdout_generators"])
    for sp in ("train", "val"):
        found = {r["generator"] for r in man[sp]} & held
        assert not found, f"{sp} contains held-out generator(s) {found}"


def test_partial_edits_absent_from_train_and_val(cfg, man):
    pe = set(cfg["routing"]["partial_edit_generators"])
    for sp in ("train", "val"):
        found = {r["generator"] for r in man[sp]} & pe
        assert not found, f"{sp} contains partial-edit generator(s) {found}"


def test_forbidden_benchmark_slices_absent_from_train_and_val(man):
    """The brief: 'Do not use the following data during training.'"""
    for sp in ("train", "val"):
        bad = [r for r in man[sp]
               if "val2017" in (r.get("orig", "") or "").lower()
               or "dalle3" in (r.get("orig", "") or "").lower()]
        assert not bad, f"{sp} contains {len(bad)} judges'-benchmark rows, e.g. {bad[0]['orig']}"


def test_train_and_val_are_class_balanced(cfg, man):
    if not cfg["invariants"]["train_val_class_balance"]:
        pytest.skip("not required")
    for sp in ("train", "val"):
        c = Counter(r["label"] for r in man[sp])
        assert c["0"] == c["1"], f"{sp} not balanced: {c['0']} real vs {c['1']} fake"


def test_every_native_size_bucket_is_balanced(cfg, man):
    lo = cfg["invariants"]["bucket_ratio_min"]
    hi = cfg["invariants"]["bucket_ratio_max"]
    for sp in ("train", "val"):
        by = defaultdict(lambda: [0, 0])
        for r in man[sp]:
            if r.get("long"):
                by[bucket(r["long"])][int(r["label"])] += 1
        for b, (nr, nf) in sorted(by.items()):
            if nr + nf < 50:
                continue
            assert nr and nf, f"{sp} bucket {b} has one class only ({nr} real / {nf} fake)"
            ratio = nr / nf
            assert lo <= ratio <= hi, f"{sp} bucket {b} ratio {ratio:.2f} outside [{lo}, {hi}]"


def test_no_source_file_appears_in_two_splits(cfg, man):
    if not cfg["invariants"]["no_file_in_two_splits"]:
        pytest.skip("not required")
    where = defaultdict(set)
    for sp in SPLITS:
        for r in man[sp]:
            key = r.get("orig") or r["path"]
            where[key].add(sp)
    bad = {k: v for k, v in where.items() if len(v) > 1}
    assert not bad, f"{len(bad)} source files span splits, e.g. {list(bad.items())[:2]}"


def test_no_subject_is_one_sided(cfg, man):
    """The bug that came back twice: bedroom 92.7:1, then 12.55:1 from the opposite fix."""
    if not cfg["invariants"]["content_two_sided"]:
        pytest.skip("not required")
    from scripts.content_audit import load_artifact_categories, subject
    cats = load_artifact_categories()
    limit = float(cfg["invariants"]["content_max_ratio"])
    for sp in ("train", "val"):
        by = defaultdict(lambda: [0, 0])
        for r in man[sp]:
            by[subject(r, cats)][int(r["label"])] += 1
        for subj, (nr, nf) in sorted(by.items()):
            if nr + nf < 200:
                continue
            assert nr and nf, f"{sp}: subject '{subj}' is one-sided ({nr} real / {nf} fake)"
            ratio = max(nr / nf, nf / nr)
            assert ratio <= limit, \
                f"{sp}: subject '{subj}' skewed {nr} real / {nf} fake (={ratio:.1f}:1, limit {limit})"


def test_cap_source_respected(cfg, man):
    for src, cap in (cfg["routing"].get("cap_source") or {}).items():
        n = sum(1 for sp in SPLITS for r in man[sp] if r["source"] == src and sp in ("train", "val"))
        assert n <= cap, f"source {src} contributes {n} train/val rows, cap is {cap}"


def test_config_and_builder_agree_on_routing(cfg):
    """The config is the source of truth; the builder must not drift from it."""
    from scripts import build_canon6 as b
    assert set(cfg["routing"]["holdout_generators"]) == set(b.HOLDOUT)
    assert set(cfg["routing"]["partial_edit_generators"]) == set(b.PARTIAL_EDIT)
    assert (cfg["routing"].get("cap_source") or {}) == b.CAP_SOURCE

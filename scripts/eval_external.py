"""Score an EXTERNAL benchmark through the PRODUCTION path, on its original files.

Thinh's rule: "benchmark and production must be absolutely decoupled". Anything shared between the
pipeline that builds the model and the one that measures it can create a correlation the model
learns and the benchmark rewards -- at which point the benchmark measures our pipeline, not the
model. So an external set is NEVER put through scripts/canonicalize.py (that is the training-data
transformation, and applying it manufactures agreement). It is scored exactly as src/predict.py
would score a user's file: original bytes in, vote(L=320) inference, score out.

Expects a directory tree whose top-level folders name the class/generator, e.g. OmniFake val:
    <root>/real/**            -> label 0
    <root>/<Generator>/**     -> label 1, generator = folder name

    python -m scripts.eval_external --root data/omnival/data/x_val \
        --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt" --per-class 400 \
        --train data/manifests/canon6_train.csv --out outputs/pe_ft/external_omnifake.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import load_image
from src.model import load_model

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_NAMES = {"real", "reals", "nature", "0_real", "real_images"}
BATCH = 32


def trained_generators(manifest):
    if not manifest or not os.path.exists(manifest):
        return None
    with open(manifest, newline="") as fh:
        return {r["generator"].replace("_", "").lower()
                for r in csv.DictReader(fh) if r["label"] == "1" and r["generator"]}


def collect(root, per_class, seed):
    groups = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        files = []
        for dp, _, fns in os.walk(p):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in IMG:
                    files.append(os.path.join(dp, fn))
            if len(files) > per_class * 6:
                break
        if not files:
            continue
        random.Random(seed).shuffle(files)
        groups[d] = files[:per_class]
    return groups


def score(model, paths):
    out = []
    for i in range(0, len(paths), BATCH):
        imgs, keep = [], []
        for p in paths[i:i + BATCH]:
            try:
                imgs.append(load_image(p)); keep.append(p)
            except Exception:
                pass
        if imgs:
            out.extend(zip(keep, model.predict(imgs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--train", default="data/manifests/canon6_train.csv",
                    help="only to LABEL which generators we trained on; never used to tune")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    groups = collect(a.root, a.per_class, a.seed)
    real_dirs = [d for d in groups if d.lower() in REAL_NAMES]
    fake_dirs = [d for d in groups if d.lower() not in REAL_NAMES]
    if not real_dirs:
        raise SystemExit(f"no real folder found under {a.root} (top-level: {sorted(groups)[:12]}). "
                         "An external set without its OWN reals is not a benchmark.")
    print(f"{a.root}\n  real folders: {real_dirs}\n  generators:   {len(fake_dirs)}")

    model = load_model(a.model)
    trained = trained_generators(a.train)

    reals = [p for d in real_dirs for p in groups[d]]
    rs = score(model, reals)
    r_scores = np.array([s for _, s in rs])
    thr = float(np.quantile(r_scores, 1 - a.fa))
    print(f"\n  reals scored: {len(rs)}   mean P(AI) {r_scores.mean():.3f}")
    print(f"  cut-off at {a.fa*100:.3g}% false alarms on THIS set's reals: {thr:.4f}")

    print(f"\n  {'generator':22s} {'n':>5s} {'AUROC':>8s} {'caught':>8s}  seen in training?")
    print("  " + "-" * 72)
    rows, all_f = [], []
    for d in sorted(fake_dirs):
        fs = score(model, groups[d])
        if not fs:
            continue
        f_scores = np.array([s for _, s in fs])
        y = np.r_[np.zeros(len(r_scores)), np.ones(len(f_scores))]
        auc = roc_auc_score(y, np.r_[r_scores, f_scores])
        caught = float((f_scores >= thr).mean())
        key = d.replace("_", "").lower()
        seen = "unknown" if trained is None else ("TRAINED" if key in trained else "unseen")
        rows.append({"generator": d, "n": len(fs), "auroc": auc, "caught": caught, "seen": seen})
        all_f.append(f_scores)
        print(f"  {d:22s} {len(fs):5d} {auc:8.4f} {caught*100:7.1f}%  {seen}")

    f_all = np.concatenate(all_f)
    y = np.r_[np.zeros(len(r_scores)), np.ones(len(f_all))]
    pooled = roc_auc_score(y, np.r_[r_scores, f_all])
    print(f"\n  POOLED  AUROC {pooled:.4f}  ({len(r_scores)} real / {len(f_all)} fake)")
    print(f"  at {a.fa*100:.3g}% false alarms: {float((f_all >= thr).mean())*100:.1f}% of AI images caught")
    for tag in ("unseen", "TRAINED"):
        sub = [r for r in rows if r["seen"] == tag]
        if sub:
            print(f"    {tag:8s} generators: {len(sub)}  mean AUROC "
                  f"{np.mean([r['auroc'] for r in sub]):.4f}  mean caught "
                  f"{np.mean([r['caught'] for r in sub])*100:.1f}%")
    print("\n  Scored on ORIGINAL files through the production path — this set never went through "
          "scripts/canonicalize.py, and nothing was tuned on it.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"root": a.root, "model": a.model, "threshold": thr, "fa": a.fa,
                   "pooled_auroc": pooled, "n_real": len(r_scores), "n_fake": len(f_all),
                   "per_generator": rows}, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

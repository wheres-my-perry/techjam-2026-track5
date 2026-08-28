"""Fit the stacker on member scores over (augmented) validation images.

    python -m src.approaches.stacked.train --val data/manifests/wildfake_val.csv \
        --members "patch_relation:...,vote+resnet_ft:...,clip_linear:...,real_manifold:..." \
        --limit 2500 --aug-views 2 --out outputs/stacked/baseline.npz

Never touches train images (members already saw those) and never touches test
or official manifests. Each val image contributes a clean copy plus
--aug-views contest-transformed copies, so the stacker is fit on the same
input distribution the eval grid produces. Fits logistic regression AND
gradient boosting; keeps whichever wins on a held-out fifth of the images.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time

import numpy as np

from src.data import load_image, load_manifest
from src.metrics import auroc
from src.model import load_model
from src.transforms import random_train_transform


def build_views(samples, aug_views, seed):
    rng = random.Random(seed)
    imgs, ys, owner = [], [], []
    for i, s in enumerate(samples):
        try:
            img = load_image(s.path)
        except Exception as e:
            print(f"skip {s.path}: {e}", flush=True)
            continue
        views = [img] + [random_train_transform(img, rng)
                         for _ in range(aug_views)]
        for v in views:
            imgs.append(v)
            ys.append(s.label)
            owner.append(i)
    return imgs, np.asarray(ys, dtype=np.int64), np.asarray(owner)


def member_scores(spec, imgs, batch=64):
    model = load_model(spec)
    out = np.zeros(len(imgs), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(imgs), batch):
        out[i:i + batch] = model.predict(imgs[i:i + batch])
        if (i // batch) % 20 == 0:
            done = min(i + batch, len(imgs))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {spec}: {done}/{len(imgs)} ({rate:.1f}/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--members", required=True,
                    help="comma-separated model specs (name[:weights])")
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--aug-views", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/stacked/baseline.npz")
    args = ap.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    print(f"members: {members}", flush=True)

    samples = load_manifest(args.val)
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.limit]
    imgs, y, owner = build_views(samples, args.aug_views, args.seed)
    print(f"{len(samples)} images -> {len(imgs)} views", flush=True)

    S = np.stack([member_scores(m, imgs) for m in members], axis=1)

    # split by IMAGE (not view) so no image leaks across fit/holdout
    n_img = owner.max() + 1
    hold_imgs = set(range(int(n_img * 0.8), n_img))
    hold = np.array([o in hold_imgs for o in owner])
    fit = ~hold

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    results = {}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(S[fit], y[fit])
    results["lr"] = (auroc(y[hold], lr.predict_proba(S[hold])[:, 1]), lr)
    hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                         random_state=args.seed)
    hgb.fit(S[fit], y[fit])
    results["hgb"] = (auroc(y[hold], hgb.predict_proba(S[hold])[:, 1]), hgb)
    for j, m in enumerate(members):
        print(f"member holdout AUROC {m}: {auroc(y[hold], S[hold, j]):.4f}",
              flush=True)
    for k, (a, _) in results.items():
        print(f"stacker {k}: holdout AUROC {a:.4f}", flush=True)

    kind = max(results, key=lambda k: results[k][0])
    best_auc, clf = results[kind]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if kind == "lr":
        np.savez(args.out, members=np.array(members), kind="lr",
                 w=clf.coef_[0], b=clf.intercept_[0])
    else:
        np.savez(args.out, members=np.array(members), kind="hgb",
                 w=np.zeros(len(members)), b=0.0)
        with open(args.out + ".pkl", "wb") as fh:
            pickle.dump(clf, fh)
    print(f"saved {kind} stacker (holdout AUROC {best_auc:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()

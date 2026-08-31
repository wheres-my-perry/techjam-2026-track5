"""Compare models on CLEAN-only, AUGMENTED-only, and the 50/50 judges' mix.

Thinh (2026-08-31): the judges score roughly 50% clean and 50% augmented, and pooling all 15
conditions is ~7% clean / 93% augmented -- so the pooled number answers neither question. If the
consistency loss trades fine detail for stability it should LOSE on clean and WIN on augmented;
the pooled number cannot see that.

    python -m scripts.slices canon6_mlp canon6_mlp_consist [--fa 0.01]
reads outputs/pe_ft/eval_<name>_official/scores.npz
"""
from __future__ import annotations
import argparse, os, numpy as np
from sklearn.metrics import roc_auc_score


def load(name):
    # A bare directory path is accepted so a specific run can be named explicitly. Otherwise the
    # _official900 directory is preferred: it is the 900-image subsample every model is compared
    # on, and picking the older _official run instead silently reintroduces the subsample-size
    # mismatch this script exists to remove. (2026-08-31)
    if os.path.isdir(name):
        f = os.path.join(name, "scores.npz")
        return (np.load(f, allow_pickle=True), f) if os.path.exists(f) else (None, None)
    for d in (f"outputs/pe_ft/eval_{name}_official900", f"outputs/pe_ft/eval_{name}_official"):
        f = os.path.join(d, "scores.npz")
        if os.path.exists(f):
            return np.load(f, allow_pickle=True), f
    return None, None


def matrix(y, s, fa):
    r, f = s[y == 0], s[y == 1]
    thr = float(np.quantile(r, 1 - fa)) if len(r) else 0.5
    tp, fn = int((f >= thr).sum()), int((f < thr).sum())
    fp, tn = int((r >= thr).sum()), int((r < thr).sum())
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
    return thr, auc, tp, fn, fp, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fa", type=float, default=0.01)
    a = ap.parse_args()

    for name in a.names:
        o, path = load(name)
        if o is None:
            print(f"{name}: no scores.npz\n"); continue
        y0 = o["labels"]
        conds = [k[6:] for k in o.files if k.startswith("score_")]
        aug = sorted(c for c in conds if c != "clean")
        clean_s, clean_y = o["score_clean"], y0
        aug_s = np.concatenate([o[f"score_{c}"] for c in aug])
        aug_y = np.tile(y0, len(aug))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(aug_s), size=len(clean_s), replace=False)
        mix_s = np.concatenate([clean_s, aug_s[pick]])
        mix_y = np.concatenate([clean_y, aug_y[pick]])

        print(f"########## {name}   ({path}, {len(aug)} transformed conditions)")
        for label, y, s in (("CLEAN only", clean_y, clean_s),
                            ("AUGMENTED only", aug_y, aug_s),
                            ("50/50 judges' mix", mix_y, mix_s)):
            thr, auc, tp, fn, fp, tn = matrix(y, s, a.fa)
            print(f"  {label:18s} n={len(s):6d} ({int((y==0).sum())} real / {int((y==1).sum())} AI)"
                  f"  AUROC {auc:.4f}   cut-off {thr:.4f}")
            print(f"      actually AI    -> AI {tp:6d}   -> real {fn:6d}")
            print(f"      actually real  -> AI {fp:6d}   -> real {tn:6d}")
            print(f"      recall {tp/max(1,tp+fn)*100:5.1f}%   false-alarm {fp/max(1,fp+tn)*100:.2f}%")
        print()


if __name__ == "__main__":
    main()

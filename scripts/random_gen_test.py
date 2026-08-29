"""Random unseen-generator test: score folders of fakes from generators never used in
training against never-trained real photos of similar native size, at native resolution
through the app policy (vote(L=320)). Prints per-generator AUROC and catch rate.

  python -m scripts.random_gen_test --root DIR --model SPEC [--n 300]
DIR holds sub-folders: real_* (label 0) and anything else (label 1, one per generator).
"""
from __future__ import annotations
import argparse, os, glob, numpy as np
from sklearn.metrics import roc_auc_score
from src.data import load_image
from src.model import load_model
from src.predict import iter_image_paths


def score_dir(m, d, n, seed=0):
    ps = sorted(iter_image_paths(d))
    rng = np.random.default_rng(seed)
    if len(ps) > n: ps = [ps[i] for i in sorted(rng.choice(len(ps), n, replace=False))]
    out, sizes = [], []
    for i in range(0, len(ps), 32):
        ims = [load_image(p) for p in ps[i:i + 32]]
        sizes += [im.size for im in ims]
        out.append(np.asarray(m.predict(ims)))
    return np.concatenate(out) if out else np.zeros(0), sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="vote(L=320)+pe_ft:outputs/pe_ft/canon3.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save", default="", help="write per-image scores to this CSV")
    a = ap.parse_args()
    m = load_model(a.model)
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d))
    reals, fakes = {}, {}
    for d in dirs:
        s, sz = score_dir(m, d, a.n)
        if len(s) == 0: continue
        (reals if os.path.basename(d).startswith("real_") else fakes)[os.path.basename(d)] = (s, sz)
    r_all = np.concatenate([s for s, _ in reals.values()])
    if a.save:
        import csv
        with open(a.save, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["set", "label", "score", "w", "h"])
            for k, (s, sz) in list(reals.items()) + list(fakes.items()):
                for v, (wd, ht) in zip(s, sz): w.writerow([k, 0 if k.startswith("real_") else 1, f"{v:.5f}", wd, ht])
    print(f"model {a.model}")
    print(f"{'set':28s} {'n':>4s} {'median':>7s} {'mean':>6s} {'>=thr':>6s} {'AUROC vs reals':>15s}  typical size")
    for k, (s, sz) in reals.items():
        print(f"{k:28s} {len(s):4d} {np.median(s):7.3f} {s.mean():6.3f} {(s >= a.threshold).mean():6.2f} {'-':>15s}  {max(set(sz), key=sz.count)}")
    for k, (s, sz) in fakes.items():
        y = np.r_[np.zeros(len(r_all)), np.ones(len(s))]
        auc = roc_auc_score(y, np.r_[r_all, s])
        print(f"{k:28s} {len(s):4d} {np.median(s):7.3f} {s.mean():6.3f} {(s >= a.threshold).mean():6.2f} {auc:15.3f}  {max(set(sz), key=sz.count)}")
    print(f"reals pooled: n={len(r_all)}  false-alarm rate at {a.threshold}: {(r_all >= a.threshold).mean():.3f}")
    f_all = np.concatenate([s for s, _ in fakes.values()])
    y = np.r_[np.zeros(len(r_all)), np.ones(len(f_all))]; sc = np.r_[r_all, f_all]
    print(f"POOLED all {len(fakes)} generators ({len(f_all)} fakes) vs {len(r_all)} reals: AUROC {roc_auc_score(y, sc):.4f}  "
          f"catch@{a.threshold} {(f_all >= a.threshold).mean():.3f}  FA@{a.threshold} {(r_all >= a.threshold).mean():.3f}")
    for fa in (0.01, 0.05):
        thr = np.quantile(r_all, 1 - fa)
        print(f"  at {fa:.0%} false alarms (thr {thr:.3f}): overall catch {(f_all >= thr).mean():.3f}; worst generator "
              f"{min(fakes, key=lambda k: (fakes[k][0] >= thr).mean())} {min((fakes[k][0] >= thr).mean() for k in fakes):.2f}")


if __name__ == "__main__":
    main()

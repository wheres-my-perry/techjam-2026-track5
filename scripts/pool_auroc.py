"""One ranking over EVERY evaluation group (Thinh, 2026-08-30): pooled AUROC + fixed-cut-off
catch / false-alarm, with each group's weight (images / total) stated.

  python -m scripts.pool_auroc --official outputs/pe_ft/eval_X_official/scores.npz \
      --unseen outputs/random_gen/X_scores_full.csv [--test outputs/pe_ft/eval_X_test/scores.npz] \
      [--wild wild.csv] [--threshold 0.15] [--save pool.csv]

Two pools are printed: clean images only, and clean + the 14 corruptions of every scores.npz
(each corrupted copy counts as one more image). Groups are unequal in size; the weight column is
there so nobody mistakes the pool for an equal-weight average.
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

WILD_DEFAULT = None  # pass --wild CSV with columns label,score


def load_npz(path, group):
    o = np.load(path); y = o["labels"]
    return pd.concat([pd.DataFrame({"group": group, "cond": k[6:], "label": y, "score": o[k]})
                      for k in o.files if k.startswith("score_")], ignore_index=True)


def line(name, df, thr):
    nf = int((df.label == 1).sum()); nr = len(df) - nf
    auc = roc_auc_score(df.label, df.score) if nf and nr else float("nan")
    return dict(group=name, images=len(df), fakes=nf, reals=nr, auroc=auc,
                caught=(df.score[df.label == 1] >= thr).mean() * 100 if nf else float("nan"),
                flagged=(df.score[df.label == 0] >= thr).mean() * 100 if nr else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--wild", default="")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    parts = [load_npz(a.official, "judges' benchmark (DALL-E-3 vs COCO val)")]
    if a.test:
        parts.append(load_npz(a.test, "canon4_test (held-out, 32 known generators incl. tampering)"))
    u = pd.read_csv(a.unseen)
    parts.append(pd.DataFrame({"group": "64 unseen generators vs 900 unseen reals (native size)", "cond": "clean",
                               "label": u.label, "score": u.score}))
    if a.wild:
        w = pd.read_csv(a.wild)
        parts.append(pd.DataFrame({"group": "wild (iPhone photos + Gemini)", "cond": "clean", "label": w.label, "score": w.score}))
    d = pd.concat(parts, ignore_index=True)
    for title, sub in [("POOL 1 - clean images only", d[d.cond == "clean"]),
                       ("POOL 2 - clean + all corruptions of the benchmark sets", d)]:
        print(f"\n=== {title}: {len(sub)} rows, cut-off {a.threshold}")
        out = [line(g, x, a.threshold) for g, x in sub.groupby("group", sort=False)] + [line("POOLED (one ranking over all groups)", sub, a.threshold)]
        tot = len(sub)
        print(f"{'group':62s} {'images':>7s} {'weight':>7s} {'fakes':>6s} {'reals':>6s} {'AUROC':>7s} {'caught':>7s} {'flagged':>8s}")
        for r in out:
            print(f"{r['group']:62s} {r['images']:7d} {r['images'] / tot * 100:6.1f}% {r['fakes']:6d} {r['reals']:6d} {r['auroc']:7.4f} {r['caught']:6.1f}% {r['flagged']:7.1f}%")
    if a.save: d.to_csv(a.save, index=False)


if __name__ == "__main__":
    main()

"""One card per candidate model, every number read at ONE cut-off chosen by ONE rule
(1% false alarms on the 900 never-trained reals of the unseen set), plus the same at a given
fixed line. Reads the files run_B.sh / run_consist.sh produce.

  python -m scripts.model_card canon4 canon4_rw4 ... [--fa 0.01] [--fixed 0.15]
"""
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def npz_stats(path, thr):
    if not os.path.exists(path): return None
    o = np.load(path); y = o["labels"]; conds = [k[6:] for k in o.files if k.startswith("score_")]
    tf = [c for c in conds if c != "clean"]
    fc = {c: (o[f"score_{c}"][y == 1] >= thr).mean() for c in conds}
    rf = {c: (o[f"score_{c}"][y == 0] >= thr).mean() for c in conds}
    return dict(clean_f=fc["clean"], clean_r=rf["clean"], tf_f=np.mean([fc[c] for c in tf]),
                tf_r=np.mean([rf[c] for c in tf]), worst_r=max(rf[c] for c in tf), worst_f=min(fc[c] for c in tf),
                auc_clean=roc_auc_score(y, o["score_clean"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--fixed", type=float, default=0.15)
    a = ap.parse_args()
    print(f"{'model':12s} {'cut-off':>7s} | {'unseen-64 clean: caught / flagged / AUROC':>40s} | {'DALL-E: clean caught/flagged':>28s} | {'DALL-E corrupted: caught / flagged mean / worst':>46s} | {'unseen corrupted probe: caught / flagged mean / worst':>50s}")
    for name in a.names:
        u = pd.read_csv(f"outputs/random_gen/{name}_scores_full.csv"); ur = u[u.label == 0].score.values; uf = u[u.label == 1].score.values
        for label, thr in (("rule", float(np.quantile(ur, 1 - a.fa))), ("fixed", a.fixed)):
            off = npz_stats(f"outputs/pe_ft/eval_{name}_official/scores.npz", thr) or npz_stats(f"outputs/pe_ft/eval_{name}_official_t015/scores.npz", thr)
            pr = npz_stats(f"outputs/pe_ft/eval_{name}_unseen_tf/scores.npz", thr)
            s = f"{name:12s} {thr:7.3f} | {(uf >= thr).mean()*100:6.1f}% / {(ur >= thr).mean()*100:4.1f}% / {roc_auc_score(u.label, u.score):.4f}{'':17s}"
            s += f" | {off['clean_f']*100:5.1f}% / {off['clean_r']*100:4.1f}%{'':15s}" if off else " | (no official eval)"
            s += f" | {off['tf_f']*100:5.1f}% / {off['tf_r']*100:4.1f}% / {off['worst_r']*100:4.1f}%{'':24s}" if off else " |"
            s += f" | {pr['tf_f']*100:5.1f}% / {pr['tf_r']*100:4.1f}% / {pr['worst_r']*100:4.1f}%" if pr else " | (no probe)"
            print(s + ("   <- " + label))
        print()


if __name__ == "__main__":
    main()

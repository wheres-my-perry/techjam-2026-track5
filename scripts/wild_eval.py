"""Score the frozen HACK SET (data/hack: phone photos + modern-generator images,
never trained on) with any model spec.

NAME: this is the HACK SET, after the folder it lives in. It was called the "wild set", which
collides with WildFake -- one of our actual datasets -- and was ambiguous in every report. Reports per-file scores, accuracy
at 0.5 and AUROC. This is the number that reflects the demo, not the benchmark.

    python -m scripts.wild_eval --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt"
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import load_image
from src.model import load_model
from src.transforms import EVAL_GRID, EXTRA_GRID

# data/hack/<group>/*  -- label 0 = authentic photo, 1 = AI-generated.
# "DALL E" (8 Bing/DALL-E images) was on disk but not in this dict, so it was being
# scored by nobody; added 2026-08-31 so the demo number covers every wild image we hold.
GROUPS = {"real": 0, "gemini": 1, "DALL E": 1}
LOOSE = {}  # the two loose files are duplicates of gemini_1/gemini_2 (identical scores 2026-08-29)


def files():
    out = []
    for g, lab in GROUPS.items():
        for f in sorted(glob.glob(f"data/hack/{g}/*")):
            if not os.path.basename(f).startswith("."):
                out.append((f, lab, g))
    for f, lab in LOOSE.items():
        if os.path.exists("data/hack/" + f):
            out.append(("data/hack/" + f, lab, "loose"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quiet", action="store_true")
    # 2026-08-31 (Thinh: "you didn't augment the hack"). This script scored the 25 files CLEAN and
    # the resulting AUROC was being used to rank models on ROBUSTNESS, which it cannot see. --grid
    # re-scores every file under the contest transform grid and pools all conditions into one set,
    # the same way src.evaluate + scripts.confusion report the judges' set.
    ap.add_argument("--grid", action="store_true",
                    help="score every file under all 15 contest conditions and pool them")
    ap.add_argument("--extra", action="store_true",
                    help="with --grid, also include the stacked conditions (chain_repost, "
                         "stack2..stack6)")
    args = ap.parse_args()
    m = load_model(args.model)
    fs = files()
    base = [load_image(f) for f, _, _ in fs]
    ylab = np.array([lab for _, lab, _ in fs])

    grid = ([("clean", lambda im: im)] if not args.grid else
            list(EVAL_GRID) + (list(EXTRA_GRID) if args.extra else []))
    per_cond = {}
    for name, fn in grid:
        per_cond[name] = m.predict([fn(im) for im in base])

    if args.grid:
        pooled = np.concatenate([per_cond[n] for n, _ in grid])
        ypool = np.tile(ylab, len(grid))
        auc = roc_auc_score(ypool, pooled) if len(set(ypool)) == 2 else float("nan")
        print(f"HACK SET POOLED over {len(grid)} conditions  n={len(pooled)} "
              f"({int((ypool==0).sum())} real / {int((ypool==1).sum())} AI)  AUROC={auc:.4f}")
        # one global cut-off, 1% false alarms on the pooled reals -- same rule as the judges' set
        r = pooled[ypool == 0]
        thr = float(np.quantile(r, 0.99)) if len(r) else 0.5
        f = pooled[ypool == 1]
        tp, fn_ = int((f >= thr).sum()), int((f < thr).sum())
        fp, tn = int((r >= thr).sum()), int((r < thr).sum())
        print(f"ONE GLOBAL CUT-OFF = {thr:.4f}  (at 1% false alarms on all reals)")
        print("                       predicted AI    predicted real")
        print(f"  actually AI          {tp:8d}          {fn_:8d}")
        print(f"  actually real        {fp:8d}          {tn:8d}")
        print(f"  recall {tp/max(1,tp+fn_)*100:.1f}%   false-alarm {fp/max(1,fp+tn)*100:.1f}%")
        print("  CAVEAT: only 5 real files, so the false-alarm rate here has huge error bars; "
              "read the recall column and the per-condition table, not this cut-off, as evidence.")
        print("  per condition (AUROC | mean P real | mean P fake):")
        for n, _ in grid:
            sc = per_cond[n]
            a = roc_auc_score(ylab, sc) if len(set(ylab)) == 2 else float("nan")
            print(f"    {n:16s} {a:.3f} | {sc[ylab==0].mean():.2f} | {sc[ylab==1].mean():.2f}")
        print(f"  [{args.model}]")
        return

    p = per_cond["clean"]
    ims = base
    y = ylab
    if not args.quiet:
        for (f, lab, g), im, s in zip(fs, ims, p):
            ok = "ok " if (s >= 0.5) == bool(lab) else "XX "
            print(f"  {ok} {g:7s} {os.path.basename(f)[:34]:34s} {str(im.size):13s} P(AI)={s:.3f}")
    acc = float(np.mean((p >= 0.5) == y.astype(bool)))
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    print(f"HACK SET  n={len(fs)} (real {int((y==0).sum())}, fake {int((y==1).sum())})  "
          f"CLEAN ONLY (pass --grid for the transform grid)  acc@0.5={acc:.2f}  AUROC={auc:.3f}  real mean P={p[y==0].mean():.2f}  fake mean P={p[y==1].mean():.2f}  [{args.model}]")


if __name__ == "__main__":
    main()

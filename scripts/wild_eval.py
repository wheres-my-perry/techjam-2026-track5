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
    args = ap.parse_args()
    m = load_model(args.model)
    fs = files()
    ims = [load_image(f) for f, _, _ in fs]
    p = m.predict(ims)
    y = np.array([lab for _, lab, _ in fs])
    if not args.quiet:
        for (f, lab, g), im, s in zip(fs, ims, p):
            ok = "ok " if (s >= 0.5) == bool(lab) else "XX "
            print(f"  {ok} {g:7s} {os.path.basename(f)[:34]:34s} {str(im.size):13s} P(AI)={s:.3f}")
    acc = float(np.mean((p >= 0.5) == y.astype(bool)))
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    print(f"HACK SET  n={len(fs)} (real {int((y==0).sum())}, fake {int((y==1).sum())})  "
          f"acc@0.5={acc:.2f}  AUROC={auc:.3f}  real mean P={p[y==0].mean():.2f}  fake mean P={p[y==1].mean():.2f}  [{args.model}]")


if __name__ == "__main__":
    main()

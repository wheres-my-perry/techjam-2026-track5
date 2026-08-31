"""Choose the product cut-off from a scores.npz, and print the robustness table.

canon4's shipped 0.15 was chosen at 1% false alarms on the 64-source unseen set.
That set cannot be rebuilt (its sources are documented by category only and
extract_randtest.py is not in the repo), so re-using 0.15 would be quoting an
unverifiable number. This picks the cut-off by the SAME RULE on data we do have:
the target false-alarm rate on CLEAN reals of the given evaluation set.

    python -m scripts.pick_cutoff --npz outputs/pe_ft/eval_canon6_test/scores.npz \
        [--fa 0.01] [--also 0.15] [--md robustness.md]

caught  = fraction of FAKES at or above the cut-off (recall)
flagged = fraction of REALS at or above the cut-off (false alarms)
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

# the brief's 15 single transforms; anything else in the npz is a stacked condition
BRIEF = {"clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
         "blur_s0.5", "blur_s1.0", "blur_s2.0", "resize_0.5x", "resize_0.25x",
         "noise_s0.02", "noise_s0.05", "noise_s0.10", "jitter_20", "crop_80"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fa", type=float, default=0.01, help="target false-alarm rate on clean reals")
    ap.add_argument("--also", type=float, default=None, help="also report at this fixed cut-off")
    ap.add_argument("--md", default=None, help="write a markdown table here")
    a = ap.parse_args()

    o = np.load(a.npz)
    y = o["labels"]
    conds = [k[len("score_"):] for k in o.files if k.startswith("score_")]
    conds.sort(key=lambda c: (c != "clean", c not in BRIEF, c))
    if "score_clean" not in o.files:
        raise SystemExit("npz has no clean condition; cannot choose a cut-off")

    real_clean = o["score_clean"][y == 0]
    thr = float(np.quantile(real_clean, 1 - a.fa))
    print(f"{a.npz}\n  n = {len(y)} ({int((y==0).sum())} real / {int((y==1).sum())} fake)")
    print(f"  cut-off @ {a.fa*100:.3g}% false alarms on CLEAN reals = {thr:.4f}\n")

    lines = ["| condition | in brief | AUROC | caught (fakes) | flagged (reals) |",
             "|---|---|---|---|---|"]
    rows = []
    for c in conds:
        s = o[f"score_{c}"]
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
        caught = float((s[y == 1] >= thr).mean())
        flagged = float((s[y == 0] >= thr).mean())
        rows.append((c, auc, caught, flagged))
        lines.append(f"| {c} | {'yes' if c in BRIEF else 'stacked'} | {auc:.4f} | "
                     f"{caught*100:.1f}% | {flagged*100:.1f}% |")
        print(f"  {c:16s} {'brief  ' if c in BRIEF else 'stacked'} AUROC {auc:.4f}   "
              f"caught {caught*100:5.1f}%   flagged {flagged*100:5.1f}%")

    tf = [r for r in rows if r[0] != "clean"]
    brief_tf = [r for r in tf if r[0] in BRIEF]
    stack_tf = [r for r in tf if r[0] not in BRIEF]
    clean = [r for r in rows if r[0] == "clean"][0]

    def summarize(tag, group):
        if not group:
            return
        print(f"\n  {tag}: AUROC mean {np.mean([r[1] for r in group]):.4f} "
              f"worst {min(r[1] for r in group):.4f} ({min(group, key=lambda r: r[1])[0]}) | "
              f"caught mean {np.mean([r[2] for r in group])*100:.1f}% | "
              f"flagged mean {np.mean([r[3] for r in group])*100:.1f}% "
              f"worst {max(r[3] for r in group)*100:.1f}%")

    print(f"\n  clean: AUROC {clean[1]:.4f} caught {clean[2]*100:.1f}% flagged {clean[3]*100:.1f}%")
    summarize("brief single transforms (14)", brief_tf)
    summarize("stacked conditions", stack_tf)

    if a.also is not None:
        print(f"\n  at the fixed cut-off {a.also}:")
        for c in ("clean",):
            s = o[f"score_{c}"]
            print(f"    {c}: caught {(s[y==1]>=a.also).mean()*100:.1f}%  "
                  f"flagged {(s[y==0]>=a.also).mean()*100:.1f}%")
        allf = np.mean([(o[f'score_{c}'][y == 1] >= a.also).mean() for c, *_ in tf])
        allr = np.mean([(o[f'score_{c}'][y == 0] >= a.also).mean() for c, *_ in tf])
        print(f"    mean over transforms: caught {allf*100:.1f}%  flagged {allr*100:.1f}%")

    if a.md:
        hdr = (f"Cut-off {thr:.4f}, chosen at {a.fa*100:.3g}% false alarms on clean reals of "
               f"`{a.npz}` (n={len(y)}: {int((y==0).sum())} real / {int((y==1).sum())} fake).\n\n")
        open(a.md, "w").write(hdr + "\n".join(lines) + "\n")
        print(f"\n  wrote {a.md}")


if __name__ == "__main__":
    main()

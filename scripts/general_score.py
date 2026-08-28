"""ONE truthful number per iteration: performance on data the model has NOT seen.

    python -m scripts.general_score --test outputs/pe_ft/eval_canon2_test \
        --official outputs/pe_ft/eval_canon2_official

GENERAL = mean of two out-of-distribution, transform-averaged AUROCs:
  * ddpm holdout, mean over the 15-condition grid   (a generator never trained on)
  * official benchmark, mean over the grid          (the contest's DALL-E vs COCO)

In-domain numbers (canon2 clean, seen generators) are printed for reference
but labelled as such -- they are NOT the score. Standing caveats are printed
every time so they cannot be forgotten: canon2 audits are "mild", and the
official set fails the colour canary (DALL-E palette != COCO).
"""

from __future__ import annotations

import argparse
import json
import os

TAMPERED = ("lama", "mat", "generative_inpainting", "palette")


def load(d):
    with open(os.path.join(d, "results.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="eval dir on canon2_test")
    ap.add_argument("--official", required=True, help="eval dir on canon_official")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    t, o = load(args.test), load(args.official)
    pg = t["per_generator"]
    ddpm = pg.get("ddpm")
    if not ddpm:
        raise SystemExit("no ddpm row in test eval -- holdout missing?")
    general = (ddpm["mean_transformed_auroc"] + o["summary"]["mean_transformed_auroc"]) / 2

    name = args.label or t["model"]
    print(f"\n=== GENERAL SCORE  {name}: {general:.3f} ===")
    print("  (mean of ddpm-holdout mean-TF and official mean-TF; both unseen)\n")
    print(f"  ddpm holdout   clean {ddpm['clean_auroc']:.3f}  mean-TF "
          f"{ddpm['mean_transformed_auroc']:.3f}  worst {ddpm['worst_transformed_auroc']:.3f}"
          f"  (n={ddpm['n_fake']})")
    s = o["summary"]
    print(f"  official       clean {s['clean_auroc']:.3f}  mean-TF "
          f"{s['mean_transformed_auroc']:.3f}  worst {s['worst_transformed_auroc']:.3f}"
          f"  ({s['worst_condition']})")
    tam = [pg[g]["clean_auroc"] for g in TAMPERED if g in pg]
    if tam:
        print(f"  tampered       clean mean {sum(tam)/len(tam):.3f}  "
              f"(stress-test only, {len(tam)} inpainting generators)")
    s = t["summary"]
    print(f"  [in-domain]    canon2 clean {s['clean_auroc']:.3f}  mean-TF "
          f"{s['mean_transformed_auroc']:.3f}  worst {s['worst_transformed_auroc']:.3f}"
          f"   <- seen generators, NOT the score")
    hot = [(g, v["clean_auroc"], v["n_fake"]) for g, v in pg.items()
           if v["clean_auroc"] >= 0.99]
    if hot:
        print("  >=0.99 rows (shortcut-hunt before quoting): " +
              ", ".join(f"{g} {a:.3f} n={n}" for g, a, n in hot))
    print("\n  caveats: canon2 metadata/canary audits are MILD (0.55-0.65); official "
          "FAILS the colour canary (0.755) -- part of any official number is palette.")


if __name__ == "__main__":
    main()

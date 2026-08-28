"""Leave-one-SCHOOL-out manifests: the real generalization test.

    python -m scripts.family_split --holdout diffusion

`ddpm` is a weak holdout: its siblings (ddim, latent_diffusion, glide,
stable_diffusion, vq_diffusion, palette...) are all in training, so a high
ddpm row can be within-school transfer rather than generalization. This
script drops an ENTIRE school from train+val (fakes only; reals untouched)
and leaves it in test, so the test row is a school the model has never seen.
"""

from __future__ import annotations

import argparse
import csv

SCHOOLS = {
    "diffusion": {"ddpm", "ddim", "latent_diffusion", "stable_diffusion", "glide",
                  "palette", "vq_diffusion", "denoising_diffusion_gan",
                  "diffusion_gan"},
    "gan": {"biggan", "big_gan", "stargan", "star_gan", "stylegan", "stylegan1",
            "stylegan2", "stylegan3", "pro_gan", "projected_gan", "gansformer",
            "cips", "gau_gan", "cycle_gan", "face_synthetics", "sfhq"},
    "token": {"vqvae", "taming_transformer"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=sorted(SCHOOLS))
    ap.add_argument("--in-prefix", default="data/manifests/canon2")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    hold = SCHOOLS[args.holdout]
    out = args.out_prefix or f"{args.in_prefix}_no{args.holdout}"
    for split in ("train", "val", "test"):
        with open(f"{args.in_prefix}_{split}.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
            names = list(rows[0])
        keep = rows if split == "test" else \
            [r for r in rows if not (r["label"] == "1" and r["generator"] in hold)]
        with open(f"{out}_{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
            w.writeheader(); w.writerows(keep)
        nf = sum(1 for r in keep if r["label"] == "1")
        print(f"{split}: {len(rows)} -> {len(keep)} rows ({nf} fake) -> "
              f"{out}_{split}.csv", flush=True)


if __name__ == "__main__":
    main()

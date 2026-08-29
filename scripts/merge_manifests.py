"""Merge canonicalized component manifests into canon2_{train,val,test}.csv.

Policy: existing WildFake canon splits stay in their splits; ArtiFact splits
80/10/10; LSUN church 30/10/60 and LSUN bedroom 55/7/38, mirroring where the
256px fake churches/bedrooms sit (see comment below).
"""

from __future__ import annotations

import argparse
import csv
import random


TAMPERED = {"lama", "mat", "generative_inpainting", "palette"}

# Official benchmark slices. Standing rule: never train on these. ArtiFact
# ships COCO val2017 under Real/coco (4998 images -- the same slice, by name
# and by count, that official_v2 uses for its real class), so those rows must
# not reach train/val even though a dhash comparison could not prove the
# individual images are byte-identical (ArtiFact re-encodes to 200x200).
OFFICIAL_SLICES = ("/coco2017/val2017/",)


def official_slice(r) -> bool:
    return any(k in r.get("orig", "") for k in OFFICIAL_SLICES)


def tampered(r) -> bool:
    """Inpainting generators, plus GLIDE's inpainting subset (glide-in)."""
    return r.get("generator") in TAMPERED or "/glide-in/" in r.get("orig", "")


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def split(rows, fracs, seed):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    a = int(n * fracs[0])
    b = a + int(n * fracs[1])
    return rows[:a], rows[a:b], rows[b:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="data/manifests/canon2")
    ap.add_argument("--canon-suffix", default="",
                    help="read canon<suffix>_*.csv inputs (e.g. _eq for the "
                         "JPEG-history-equalized build)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = read(f"data/manifests/canon{args.canon_suffix}_wf_train.csv")
    va = read(f"data/manifests/canon{args.canon_suffix}_wf_val.csv")
    te = read(f"data/manifests/canon{args.canon_suffix}_wf_test.csv")
    # ddpm is the designated held-out generator (WildFake's 20K ddpm is
    # test-only). ArtiFact ships its own small ddpm folder; an 80/10/10 split
    # would put it in TRAIN and silently contaminate the one number we quote
    # as "unseen generator". Route every ArtiFact ddpm row to test.
    # Tampered = a real photo with a locally regenerated region (inpainting).
    # Canonical protocol (PROGRESS 2026-08-28): tampered is EXCLUDED from
    # training -- a random crop can land on the untouched part and carry a
    # "fake" label, which is label noise. They stay in test as a stress-test.
    art = read(f"data/manifests/canon{args.canon_suffix}_artifact.csv")
    n_off = sum(1 for r in art if official_slice(r))
    art = [r for r in art if not official_slice(r)]
    print(f"official-slice guard: dropped {n_off} ArtiFact COCO val2017 rows")
    art_hold = [r for r in art if r.get("generator") == "ddpm" or tampered(r)]
    art_rest = [r for r in art if not (r.get("generator") == "ddpm" or tampered(r))]
    a_tr, a_va, a_te = split(art_rest, (0.8, 0.1), args.seed)
    a_te += art_hold
    n_ddpm = sum(1 for r in art_hold if r.get("generator") == "ddpm")
    print(f"test-only routing: {n_ddpm} ArtiFact ddpm (held-out family) + "
          f"{len(art_hold) - n_ddpm} tampered/inpainting rows -> test")
    # 256-bucket reals must MIRROR the 256px fake content per split (rule in
    # project-conventions): WildFake ddim/ddpm fakes are bedrooms + churches.
    # Fake bedrooms: train 12.2K / val 1.5K / test 8.6K -> reals 55/7/38.
    # Fake churches: test-only 8.7K (ddpm held out) -> reals 30/10/60, so
    # train no longer holds 27K always-real churches ("church = real").
    c_tr, c_va, c_te = split(read(f"data/manifests/canon{args.canon_suffix}_lsun.csv"),
                             (0.3, 0.1), args.seed + 1)
    b_tr, b_va, b_te = split(read(f"data/manifests/canon{args.canon_suffix}_lsun_bedroom.csv"),
                             (0.55, 0.07), args.seed + 2)
    outs = {"train": tr + a_tr + c_tr + b_tr, "val": va + a_va + c_va + b_va,
            "test": te + a_te + c_te + b_te}
    try:  # optional hard-fake selfgen set: TRAIN ONLY (training medicine, not measurement)
        sg = read(f"data/manifests/canon{args.canon_suffix}_selfgen.csv")
        outs["train"] += sg
        print(f"included {len(sg)} selfgen hard fakes in train")
    except FileNotFoundError:
        pass
    for name, rows in outs.items():
        random.Random(args.seed + hash(name) % 1000).shuffle(rows)
        p = f"{args.out_prefix}_{name}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "orig", "label",
                                               "generator", "source"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        n_real = sum(1 for r in rows if r["label"] == "0")
        print(f"{name}: {len(rows)} rows ({n_real} real / "
              f"{len(rows)-n_real} fake) -> {p}")


if __name__ == "__main__":
    main()

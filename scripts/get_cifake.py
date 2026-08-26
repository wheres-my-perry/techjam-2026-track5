"""Download CIFAKE (HF mirror) and materialize a balanced subset to disk + manifests.

Run on a machine with internet (not the sandbox):
    pip install -r requirements-train.txt
    python scripts/get_cifake.py [--train-n 20000] [--val-n 2000] [--test-n 4000]

Creates:
    data/cifake/images/{train,val,test}/<idx>_<label>.png
    data/manifests/cifake_{train,val,test}.csv     (path,label,generator,source)

Label convention everywhere in this repo: 1 = AI-generated, 0 = real.
The HF mirror's own label names are read at runtime and mapped — never assumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_map_from_names(names):
    """Map dataset class indices -> our convention (1=fake/AI, 0=real)."""
    mapping = {}
    for idx, name in enumerate(names):
        n = str(name).lower()
        if "fake" in n or "ai" in n or "synthetic" in n:
            mapping[idx] = 1
        elif "real" in n:
            mapping[idx] = 0
        else:
            raise ValueError(f"Cannot interpret label name {name!r}")
    assert set(mapping.values()) == {0, 1}, f"Bad mapping from {names}"
    return mapping


def materialize(ds, mapping, out_dir, n_per_class, rng):
    """Write a balanced sample of `ds` to out_dir; return manifest rows."""
    os.makedirs(out_dir, exist_ok=True)
    by_class = {0: [], 1: []}
    order = list(range(len(ds)))
    rng.shuffle(order)
    labels = ds["label"]  # column access, fast
    for i in order:
        lab = mapping[labels[i]]
        if len(by_class[lab]) < n_per_class:
            by_class[lab].append(i)
        if all(len(v) >= n_per_class for v in by_class.values()):
            break
    rows = []
    for lab, idxs in by_class.items():
        for i in idxs:
            img = ds[i]["image"].convert("RGB")
            p = os.path.join(out_dir, f"{i:06d}_{lab}.png")
            img.save(p)
            rows.append({
                "path": os.path.relpath(p, ROOT),
                "label": lab,
                "generator": "sd1.4" if lab == 1 else "",
                "source": "cifake",
            })
    rng.shuffle(rows)
    return rows


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=20000, help="total train images (balanced)")
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--test-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset  # import here so --help works without deps

    print("Loading dragonintelligence/CIFAKE-image-dataset ...")
    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    names = ds["train"].features["label"].names
    mapping = label_map_from_names(names)
    print(f"Label names {names} -> ours(1=AI): {mapping}")

    rng = random.Random(args.seed)
    img_root = os.path.join(ROOT, "data", "cifake", "images")
    man_root = os.path.join(ROOT, "data", "manifests")

    # train/val carved from HF train split; test from HF test split (untouched by tuning)
    train_rows = materialize(ds["train"], mapping, os.path.join(img_root, "train"),
                             args.train_n // 2, rng)
    # val: sample from train split again but exclude picked indices via fresh shuffle on
    # remaining — simplest correct way: materialize from the *test* half of train order.
    # To keep it simple and leak-free we take val from the HF train split AFTER removing
    # train picks by filename check.
    picked = {os.path.basename(r["path"]) for r in train_rows}
    val_rows = []
    attempts = 0
    labels = ds["train"]["label"]
    order = list(range(len(ds["train"])))
    rng.shuffle(order)
    need = {0: args.val_n // 2, 1: args.val_n // 2}
    os.makedirs(os.path.join(img_root, "val"), exist_ok=True)
    for i in order:
        lab = mapping[labels[i]]
        fname = f"{i:06d}_{lab}.png"
        if fname in picked or need[lab] <= 0:
            continue
        img = ds["train"][i]["image"].convert("RGB")
        p = os.path.join(img_root, "val", fname)
        img.save(p)
        val_rows.append({"path": os.path.relpath(p, ROOT), "label": lab,
                         "generator": "sd1.4" if lab == 1 else "", "source": "cifake"})
        need[lab] -= 1
        if all(v <= 0 for v in need.values()):
            break
    rng.shuffle(val_rows)

    test_rows = materialize(ds["test"], mapping, os.path.join(img_root, "test"),
                            args.test_n // 2, rng)

    write_manifest(train_rows, os.path.join(man_root, "cifake_train.csv"))
    write_manifest(val_rows, os.path.join(man_root, "cifake_val.csv"))
    write_manifest(test_rows, os.path.join(man_root, "cifake_test.csv"))
    print("Done. DATA_ROOT is the repo root (paths in manifests are repo-relative).")


if __name__ == "__main__":
    main()

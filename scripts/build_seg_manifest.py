"""Manifest for patch-level (localisation) training on SID_Set.

Rows: path, mask (png path or empty), label (0 real / 1 fully synthetic /
2 tampered), source. Held-out TEST = images that came from SID's own
validation shards (never trained). Train/val = 90/10 of the train shards.

    python -m scripts.build_seg_manifest
"""
from __future__ import annotations

import csv
import glob
import os
import random

import pyarrow.parquet as pq

SID_LABEL = {"sid_real": 0, "sid_fake": 1, "sid_tampered": 2}


def main():
    val_ids = set()
    for shard in glob.glob("data/sid_set/data/validation-*.parquet"):
        val_ids |= set(pq.read_table(shard, columns=["img_id"]).column("img_id").to_pylist())
    print(f"held-out ids from SID validation shards: {len(val_ids)}")
    rows = []
    for r in csv.DictReader(open("data/manifests/raw_ext.csv")):
        if r["source"] not in SID_LABEL:
            continue
        img_id = os.path.splitext(os.path.basename(r["path"]))[0]
        w, h = int(r["w"]), int(r["h"])
        if min(w, h) * 448 / max(w, h) < 294:   # too narrow for the fixed 294 crop after --long 448
            continue
        lab = SID_LABEL[r["source"]]
        mask = f"data/ext/img/sid_tampered_mask/{img_id}.png" if lab == 2 else ""
        if lab == 2 and not os.path.exists(mask):
            continue
        rows.append({"path": r["path"], "mask": mask, "label": lab, "source": r["source"],
                     "split": "test" if img_id in val_ids else "trainval"})
    tv = [r for r in rows if r["split"] == "trainval"]; te = [r for r in rows if r["split"] == "test"]
    random.Random(0).shuffle(tv)
    if not te:  # validation shards not on disk: hold out 10% of train shards by image (seeded)
        n_te = len(tv) // 10; te, tv = tv[:n_te], tv[n_te:]
    n_val = len(tv) // 10
    out = {"val": tv[:n_val], "train": tv[n_val:], "test": te}
    for sp, rs in out.items():
        p = f"data/manifests/seg_{sp}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "mask", "label", "source"], extrasaction="ignore")
            w.writeheader(); w.writerows(rs)
        c = [sum(int(r["label"]) == k for r in rs) for k in (0, 1, 2)]
        print(f"{p}: {len(rs)} rows  real {c[0]} / fake {c[1]} / tampered {c[2]}")


if __name__ == "__main__":
    main()

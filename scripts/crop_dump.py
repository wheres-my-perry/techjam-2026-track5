"""Dump PER-CROP scores (with crop positions) for the random unseen-generator set, one GPU pass,
so that crop-aggregation rules can be compared offline (scripts/crop_agg.py) without re-scoring.

  python -m scripts.crop_dump --root DIR --model "vote(t=2,L=320)+pe_ft:CKPT" --save OUT.npz

Same folder convention and seeded 300-per-folder subsample as scripts.random_gen_test, so the
image set is identical across dumps. Saved arrays (one row per crop):
  img (int, image index), x0, y0, x1, y1 (box on the shrunk image), score (float32)
and per image: set (str), label (0 real / 1 fake), path, w, h (shrunk size), nw, nh (native size).
"""
from __future__ import annotations
import argparse, os, glob, numpy as np
from src.data import load_image
from src.model import load_model
from src.predict import iter_image_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    m = load_model(a.model)
    assert hasattr(m, "_boxes"), "model spec must be a vote(...) wrapper"
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d))
    sets, labels, paths, W, H, NW, NH = [], [], [], [], [], [], []
    img_idx, X0, Y0, X1, Y1, S = [], [], [], [], [], []
    for d in dirs:
        name = os.path.basename(d)
        ps = sorted(iter_image_paths(d))
        rng = np.random.default_rng(0)
        if len(ps) > a.n: ps = [ps[i] for i in sorted(rng.choice(len(ps), a.n, replace=False))]
        for i in range(0, len(ps), 32):
            batch = ps[i:i + 32]
            views, owners, boxes = [], [], []
            for p in batch:
                im = load_image(p)
                k = len(paths)
                sets.append(name); labels.append(0 if name.startswith("real_") else 1); paths.append(p)
                NW.append(im.size[0]); NH.append(im.size[1])
                sim, bx = m._boxes(im)
                W.append(sim.size[0]); H.append(sim.size[1])
                for b in bx:
                    views.append(sim.crop(b)); owners.append(k); boxes.append(b)
            sc = np.asarray(m.inner.predict(views), dtype=np.float32)
            img_idx += owners; S.append(sc)
            for b in boxes:
                X0.append(b[0]); Y0.append(b[1]); X1.append(b[2]); Y1.append(b[3])
        print(f"{name:30s} {len(ps):4d} images  {len(img_idx):8d} crops so far", flush=True)
    np.savez_compressed(a.save, model=a.model, set=np.array(sets), label=np.array(labels, np.int8),
                        path=np.array(paths), w=np.array(W), h=np.array(H), nw=np.array(NW), nh=np.array(NH),
                        img=np.array(img_idx, np.int32), x0=np.array(X0, np.int16), y0=np.array(Y0, np.int16),
                        x1=np.array(X1, np.int16), y1=np.array(Y1, np.int16), score=np.concatenate(S))
    print(f"saved {a.save}: {len(paths)} images, {len(img_idx)} crops")


if __name__ == "__main__":
    main()

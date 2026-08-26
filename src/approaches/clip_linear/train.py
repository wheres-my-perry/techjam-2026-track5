"""Train the linear head on (cached) CLIP embeddings.

    python -m src.approaches.clip_linear.train \
        --train data/manifests/wildfake_train.csv --val data/manifests/wildfake_val.csv \
        [--augment-views 2] [--backbone ViT-L-14] [--out outputs/clip_linear/baseline.pt]

Embeddings are extracted once per (manifest, backbone, augment setting) and cached
under outputs/clip_linear/cache/ — re-training the head afterwards takes seconds.
--augment-views N adds N extra augmented copies of each training image (pixel-level
contest transforms applied BEFORE CLIP preprocessing).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random

import numpy as np
import torch

from src.data import load_image, load_manifest
from src.metrics import auroc
from src.transforms import random_train_transform
from .model import embed_images, load_backbone, pick_device

CACHE_DIR = "outputs/clip_linear/cache"


def cache_key(manifest, backbone, pretrained, views, seed):
    h = hashlib.md5(f"{os.path.abspath(manifest)}|{backbone}|{pretrained}|{views}|{seed}".encode())
    return os.path.join(CACHE_DIR, h.hexdigest()[:16] + ".npz")


def extract(manifest, backbone, pretrained, augment_views, seed, device, batch):
    key = cache_key(manifest, backbone, pretrained, augment_views, seed)
    if os.path.exists(key):
        z = np.load(key)
        print(f"cache hit: {key} ({len(z['y'])} rows)")
        return z["X"], z["y"]
    model, preprocess = load_backbone(backbone, pretrained, device)
    samples = load_manifest(manifest)
    rng = random.Random(seed)
    X_parts, y_parts = [], []
    todo = []  # (image, label) pairs, materialized lazily in chunks
    def flush():
        if not todo:
            return
        imgs, ys = zip(*todo)
        X_parts.append(embed_images(model, preprocess, list(imgs), device, batch))
        y_parts.append(np.asarray(ys, dtype=np.int64))
        todo.clear()
    for n, s in enumerate(samples):
        img = load_image(s.path)
        todo.append((img, s.label))
        for _ in range(augment_views):
            todo.append((random_train_transform(img, rng), s.label))
        if len(todo) >= batch * 4:
            flush()
        if (n + 1) % 1000 == 0:
            print(f"  embedded {n + 1}/{len(samples)} images")
    flush()
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(key, X=X, y=y)
    print(f"cached {len(y)} embeddings -> {key}")
    return X, y


def train_head(Xtr, ytr, Xva, yva, epochs=200, lr=1e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    device = "cpu"  # embeddings are tiny; CPU is instant and deterministic
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = (-1.0, None, None)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(Xtr_t @ w + b, ytr_t)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs:
            with torch.no_grad():
                sva = torch.sigmoid(torch.tensor(Xva, dtype=torch.float32) @ w + b).numpy()
            va = auroc(yva, sva)
            print(f"  head epoch {ep}: loss={loss.item():.4f} val_auroc={va:.4f}")
            if va > best[0]:
                best = (va, w.detach().numpy().copy(), float(b.item()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--backbone", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--augment-views", type=int, default=0,
                    help="N augmented copies per train image (0 = clean only)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/clip_linear/baseline.pt")
    args = ap.parse_args()

    device = pick_device()
    print(f"device={device} backbone={args.backbone}/{args.pretrained} "
          f"augment_views={args.augment_views}")

    Xtr, ytr = extract(args.train, args.backbone, args.pretrained,
                       args.augment_views, args.seed, device, args.batch)
    Xva, yva = extract(args.val, args.backbone, args.pretrained, 0,
                       args.seed, device, args.batch)

    best_auc, w, b = train_head(Xtr, ytr, Xva, yva, epochs=args.epochs, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"w": w, "b": b, "backbone": args.backbone,
                "pretrained": args.pretrained, "val_auroc": best_auc,
                "augment_views": args.augment_views}, args.out)
    print(f"best val AUROC {best_auc:.4f}; saved -> {args.out}")


if __name__ == "__main__":
    main()

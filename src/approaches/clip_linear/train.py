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
SHARD = 2000  # samples per cache shard; kills lose at most one shard of work


def cache_dir_for(manifest, backbone, pretrained, views, seed):
    h = hashlib.md5(f"{os.path.abspath(manifest)}|{backbone}|{pretrained}|{views}|{seed}".encode())
    return os.path.join(CACHE_DIR, h.hexdigest()[:16])


def _embed_shard(model, preprocess, chunk, augment_views, rng, device, batch):
    X_parts, ys, todo = [], [], []

    def flush():
        if todo:
            X_parts.append(embed_images(model, preprocess, todo, device, batch))
            todo.clear()

    for s in chunk:
        img = load_image(s.path)
        todo.append(img)
        ys.append(s.label)
        for _ in range(augment_views):
            todo.append(random_train_transform(img, rng))
            ys.append(s.label)
        if len(todo) >= batch * 4:
            flush()
    flush()
    return np.concatenate(X_parts), np.asarray(ys, dtype=np.int64)


def extract(manifest, backbone, pretrained, augment_views, seed, device, batch):
    """Sharded + resumable: each SHARD-sample chunk is cached as its own .npz,
    so an interrupted run resumes from the last completed shard."""
    key_dir = cache_dir_for(manifest, backbone, pretrained, augment_views, seed)
    os.makedirs(key_dir, exist_ok=True)
    samples = load_manifest(manifest)
    chunks = [samples[i:i + SHARD] for i in range(0, len(samples), SHARD)]
    model = preprocess = None
    X_parts, y_parts, cached = [], [], 0
    for ci, chunk in enumerate(chunks):
        sp = os.path.join(key_dir, f"shard_{ci:05d}.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            X_parts.append(z["X"])
            y_parts.append(z["y"])
            cached += 1
            continue
        if model is None:  # lazy: fully-cached extraction never loads the backbone
            model, preprocess = load_backbone(backbone, pretrained, device)
        rng = random.Random(seed * 1_000_003 + ci)  # per-shard rng: resume-stable
        X, y = _embed_shard(model, preprocess, chunk, augment_views, rng, device, batch)
        tmp = sp + ".tmp.npz"
        np.savez_compressed(tmp, X=X, y=y)
        os.replace(tmp, sp)  # atomic: a kill mid-write never corrupts a shard
        X_parts.append(X)
        y_parts.append(y)
        print(f"  shard {ci + 1}/{len(chunks)} embedded ({len(y)} rows)", flush=True)
    if cached:
        print(f"  ({cached}/{len(chunks)} shards from cache)")
    return np.concatenate(X_parts), np.concatenate(y_parts)


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

"""Train the relation head on cached patch embeddings (trunk frozen).

    python -m src.approaches.patch_relation.train \
        --train data/manifests/wildfake_train.csv --val data/manifests/wildfake_val.csv \
        --trunk outputs/resnet_ft/wf_aug.pt --augment-views 2 \
        --out outputs/patch_relation/baseline.pt

Two stages, both kill-resilient:
1. Extraction (the expensive part): every image -> [clean + N augmented views]
   -> 9 grid crops each -> frozen trunk -> (9, 2048) float16, cached in
   resumable shards (atomic writes, one shard lost at most on kill).
2. Head training (cheap): 2-layer transformer over the 9 patch tokens,
   minutes per run on the cached embeddings; best-val checkpoint saved with
   the trunk weights embedded (self-contained for inference).
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
from .model import (RelationHead, grid_views, load_trunk_from_resnet_ft,
                    pad_views, pick_device, to_tensor)

CACHE_DIR = "outputs/patch_relation/cache"
SHARD = 1000  # images per cache shard
P = 9


def cache_dir_for(manifest, trunk, views, seed):
    h = hashlib.md5(
        f"{os.path.abspath(manifest)}|{os.path.abspath(trunk)}|{views}|{seed}"
        .encode())
    return os.path.join(CACHE_DIR, h.hexdigest()[:16])


@torch.no_grad()
def _embed_crops(trunk, crops, device, batch=192) -> np.ndarray:
    outs = []
    for i in range(0, len(crops), batch):
        x = to_tensor(crops[i:i + batch]).to(device)
        outs.append(trunk(x).float().cpu().numpy())
    return np.concatenate(outs)


def _extract_shard(trunk, chunk, augment_views, rng, device):
    crops, ys = [], []
    for s in chunk:
        try:
            img = load_image(s.path)
        except Exception as e:
            print(f"skip {s.path}: {e}", flush=True)
            continue
        views = [img] + [random_train_transform(img, rng)
                         for _ in range(augment_views)]
        for v in views:
            crops.extend(pad_views(grid_views(v)))
            ys.append(s.label)
    emb = _embed_crops(trunk, crops, device)          # (n*P, 2048)
    X = emb.reshape(len(ys), P, -1).astype(np.float16)
    return X, np.asarray(ys, dtype=np.int64)


def extract(manifest, trunk_path, augment_views, seed, device):
    key_dir = cache_dir_for(manifest, trunk_path, augment_views, seed)
    os.makedirs(key_dir, exist_ok=True)
    samples = load_manifest(manifest)
    chunks = [samples[i:i + SHARD] for i in range(0, len(samples), SHARD)]
    trunk = None
    X_parts, y_parts, cached = [], [], 0
    for ci, chunk in enumerate(chunks):
        sp = os.path.join(key_dir, f"shard_{ci:05d}.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            X_parts.append(z["X"])
            y_parts.append(z["y"])
            cached += 1
            continue
        if trunk is None:  # lazy: fully-cached extraction never loads the trunk
            trunk = load_trunk_from_resnet_ft(trunk_path).to(device).eval()
        rng = random.Random(seed * 1_000_003 + ci)  # per-shard rng: resume-stable
        X, y = _extract_shard(trunk, chunk, augment_views, rng, device)
        tmp = sp + ".tmp.npz"
        np.savez_compressed(tmp, X=X, y=y)
        os.replace(tmp, sp)  # atomic
        X_parts.append(X)
        y_parts.append(y)
        print(f"  shard {ci + 1}/{len(chunks)} embedded ({len(y)} rows)",
              flush=True)
    if cached:
        print(f"  ({cached}/{len(chunks)} shards from cache)", flush=True)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def train_head(Xtr, ytr, Xva, yva, device, epochs=30, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    rel = RelationHead().to(device)
    opt = torch.optim.AdamW(rel.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr)                    # fp16, stays on CPU
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    n = len(ytr_t)
    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        rel.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            x = Xtr_t[idx].float().to(device)
            y = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(rel(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        rel.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(Xva_t), 1024):
                s = torch.sigmoid(rel(Xva_t[i:i + 1024].float().to(device)))
                scores.append(s.cpu().numpy())
        va = auroc(yva, np.concatenate(scores))
        print(f"head epoch {ep}: loss={running/n:.4f} val_auroc={va:.4f}",
              flush=True)
        if va > best_auc:
            best_auc = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in rel.state_dict().items()}
    return best_auc, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--trunk", default="outputs/resnet_ft/wf_aug.pt")
    ap.add_argument("--augment-views", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/patch_relation/baseline.pt")
    args = ap.parse_args()

    device = pick_device()
    print(f"device={device} trunk={args.trunk} "
          f"augment_views={args.augment_views}", flush=True)

    Xtr, ytr = extract(args.train, args.trunk, args.augment_views,
                       args.seed, device)
    Xva, yva = extract(args.val, args.trunk, 0, args.seed, device)
    print(f"train {Xtr.shape} val {Xva.shape}", flush=True)

    best_auc, head_state = train_head(Xtr, ytr, Xva, yva, device,
                                      epochs=args.epochs, seed=args.seed)

    trunk = load_trunk_from_resnet_ft(args.trunk)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"head": head_state, "trunk": trunk.state_dict(),
                "trunk_src": args.trunk, "val_auroc": best_auc}, args.out)
    print(f"best val AUROC {best_auc:.4f}; weights -> {args.out}")


if __name__ == "__main__":
    main()

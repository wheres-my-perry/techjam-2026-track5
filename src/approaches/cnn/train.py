"""Train the simple CNN baseline on manifest data.

    python -m src.train_cnn --train data/manifests/cifake_train.csv \
        --val data/manifests/cifake_val.csv [--augment] [--epochs 5]

--augment applies random_train_transform (the contest-transform distribution) to
training images. Run once WITHOUT and once WITH to measure the robustness delta
— that comparison is the core experiment of the project.

Saves weights to outputs/cnn_baseline.pt (or --out). Prints val AUROC per epoch.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import SimpleCNN, pick_device, to_tensor
from src.data import load_image, load_manifest
from src.metrics import auroc
from src.transforms import random_train_transform


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, augment=False, seed=0):
        self.samples = load_manifest(manifest_csv)
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s.path)
        if self.augment:
            # per-item rng: deterministic per (seed, item, epoch-ish time) is overkill;
            # fresh rng per call keeps augmentation i.i.d. across epochs.
            img = random_train_transform(img, random.Random())
        x = to_tensor([img])[0]
        return x, float(s.label)


def collate(batch):
    xs, ys = zip(*batch)
    # crops can differ in size when augment includes center_crop: pad-free solution
    # is to resize crops back to the majority size — but for CIFAKE (all 32x32 and
    # crop of 32 -> 25px) we instead upsample smaller items to the batch max size.
    hs = [x.shape[1] for x in xs]
    ws = [x.shape[2] for x in xs]
    H, W = max(hs), max(ws)
    fixed = []
    for x in xs:
        if x.shape[1] != H or x.shape[2] != W:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            )[0]
        fixed.append(x)
    return torch.stack(fixed), torch.tensor(ys, dtype=torch.float32)


@torch.no_grad()
def evaluate_auroc(net, loader, device):
    net.eval()
    ys, ss = [], []
    for x, y in loader:
        logits = net(x.to(device))
        ss.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return auroc(ys, ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/cnn_baseline.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device={device} augment={args.augment}")

    train_ds = ManifestDataset(args.train, augment=args.augment, seed=args.seed)
    val_ds = ManifestDataset(args.val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    net = SimpleCNN(width=args.width).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        val_auc = evaluate_auroc(net, val_dl, device)
        print(f"epoch {epoch}: loss={running/seen:.4f} val_auroc={val_auc:.4f} "
              f"({time.time()-t0:.0f}s)")
        if val_auc > best:
            best = val_auc
            torch.save({"state_dict": net.state_dict(), "width": args.width,
                        "augment": args.augment, "val_auroc": val_auc}, args.out)
    print(f"best val AUROC {best:.4f}; weights -> {args.out}")


if __name__ == "__main__":
    main()

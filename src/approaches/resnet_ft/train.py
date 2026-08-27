"""Fine-tune pretrained ResNet-50 on manifest data. Kill-resumable (epoch state).

    python -m src.approaches.resnet_ft.train --train data/manifests/wildfake_train.csv \
        --val data/manifests/wildfake_val.csv --epochs 5 --augment --crop 224 --batch 24 \
        --out outputs/resnet_ft/wf_aug.pt

Self-contained by design (approaches never import each other); the dataset code
mirrors approaches/cnn/train.py with ImageNet normalization.
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

from src.data import load_image, load_manifest
from src.metrics import auroc
from src.transforms import random_train_transform
from .model import build_net, pick_device, to_tensor


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, augment=False, crop=224):
        self.samples = load_manifest(manifest_csv)
        self.augment = augment
        self.crop = crop

    def __len__(self):
        return len(self.samples)

    def _random_crop(self, img, rng):
        c = self.crop
        w, h = img.size
        if w < c or h < c:
            s = c / min(w, h)
            img = img.resize((max(c, int(w * s + 0.5)), max(c, int(h * s + 0.5))))
            w, h = img.size
        x = rng.randint(0, w - c)
        y = rng.randint(0, h - c)
        return img.crop((x, y, x + c, y + c))

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s.path)
        if self.augment:
            img = random_train_transform(img, random.Random())
        img = self._random_crop(img, random.Random())
        return to_tensor([img])[0], float(s.label)


@torch.no_grad()
def evaluate_auroc(net, loader, device):
    net.eval()
    ys, ss = [], []
    for x, y in loader:
        logits = net(x.to(device)).squeeze(1)
        ss.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return auroc(ys, ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4, help="low LR: fine-tuning")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/resnet_ft/baseline.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device={device} augment={args.augment} crop={args.crop}", flush=True)

    train_dl = DataLoader(ManifestDataset(args.train, args.augment, args.crop),
                          batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(ManifestDataset(args.val, False, args.crop),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers)

    net = build_net(pretrained=True).to(device)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best, start_epoch = -1.0, 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    state_path = args.out + ".state"
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        start_epoch, best = st["epoch"] + 1, st["best"]
        print(f"resumed from {state_path}: epoch {start_epoch} (best {best:.4f})",
              flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x).squeeze(1), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        val_auc = evaluate_auroc(net, val_dl, device)
        print(f"epoch {epoch}: loss={running/seen:.4f} val_auroc={val_auc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_auc > best:
            best = val_auc
            torch.save({"state_dict": net.state_dict(), "val_auroc": val_auc,
                        "augment": args.augment}, args.out)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best}, state_path)
    print(f"best val AUROC {best:.4f}; weights -> {args.out}")


if __name__ == "__main__":
    main()

"""Train pe_seg (per-patch head) on SID_Set real / synthetic / tampered+mask.

    python -m src.approaches.pe_seg.train --train data/manifests/seg_train.csv \
        --val data/manifests/seg_val.csv --epochs 3 --out outputs/pe_seg/sid.pt

Targets: per-patch soft label = mean of the mask inside the patch (tampered),
all 1 (synthetic), all 0 (real). Image label = 1 for synthetic and tampered.
Loss = patch BCE + image BCE (top-k pooled). Mask-aware random 336 crop for
tampered so the altered region is inside the crop (else the label is a lie).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data import load_image
from src.transforms import random_train_transform
from src.approaches.pe_ft.model import pick_device, to_tensor
from .model import PESegNet, prep, PATCH, LONG, TRAIN_CROP


class SegDataset(Dataset):
    def __init__(self, manifest, train: bool, seed: int = 0):
        self.rows = list(csv.DictReader(open(manifest, newline="")))
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(f"{self.seed}|{i}|{time.time() if self.train else 0}")
        im = prep(load_image(r["path"]), LONG)
        lab = int(r["label"])
        if lab == 2:
            mask = Image.open(r["mask"]).convert("L").resize(im.size, Image.BILINEAR)
        else:
            mask = Image.new("L", im.size, 255 if lab == 1 else 0)
        w, h = im.size
        c = min(TRAIN_CROP, w, h)
        c = (c // PATCH) * PATCH
        if self.train:
            best = None
            for _ in range(10):  # mask-aware: keep the crop that holds the most altered pixels
                x0, y0 = rng.randint(0, w - c), rng.randint(0, h - c)
                if lab != 2:
                    best = (x0, y0); break
                frac = np.asarray(mask.crop((x0, y0, x0 + c, y0 + c))).mean() / 255.0
                if best is None or frac > best[2]:
                    best = (x0, y0, frac)
                if frac >= 0.02:
                    break
            x0, y0 = best[0], best[1]
            if rng.random() < 0.5:
                im, mask = im.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
                x0 = w - c - x0
        else:
            x0, y0 = (w - c) // 2, (h - c) // 2
        im = im.crop((x0, y0, x0 + c, y0 + c)); mask = mask.crop((x0, y0, x0 + c, y0 + c))
        if self.train:
            im = random_train_transform(im, rng, geometry=False)  # label-neutral, size-preserving
        t = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None]
        t = F.avg_pool2d(t, PATCH)[0, 0]              # (c/14, c/14) soft patch targets
        return im, t, float(lab != 0), lab


def collate(batch):
    ims, ts, ys, labs = zip(*batch)
    return to_tensor(list(ims)), torch.stack(ts), torch.tensor(ys), torch.tensor(labs)


@torch.no_grad()
def evaluate(net, dl, device):
    net.eval()
    ys, ps, labs, pt, pp = [], [], [], [], []
    for x, t, y, lab in dl:
        x = x.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logit, pl = net(x)
        ps += torch.sigmoid(logit.float()).cpu().tolist(); ys += y.tolist(); labs += lab.tolist()
        m = lab == 2
        if m.any():
            pt.append((t[m] > 0.5).flatten().numpy()); pp.append(torch.sigmoid(pl.float()[m.to(device)]).flatten().cpu().numpy())
    ys, ps, labs = np.array(ys), np.array(ps), np.array(labs)
    out = {}
    real = labs == 0
    for k, name in ((1, "synthetic"), (2, "tampered")):
        sel = real | (labs == k)
        if (labs == k).any() and real.any():
            out[f"img_auroc_{name}_vs_real"] = roc_auc_score(ys[sel], ps[sel])
    if pt:
        pt, pp = np.concatenate(pt), np.concatenate(pp)
        if len(set(pt.tolist())) == 2:
            out["patch_auroc_tampered"] = roc_auc_score(pt, pp)
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", default="outputs/pe_seg/baseline.pt"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    tr = SegDataset(args.train, True, args.seed)
    if args.limit_train:
        random.Random(args.seed).shuffle(tr.rows); tr.rows = tr.rows[: args.limit_train]
    va = SegDataset(args.val, False)
    print(f"device={device} train={len(tr)} val={len(va)} crop={TRAIN_CROP} long={LONG}", flush=True)
    tdl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    vdl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)
    net = PESegNet(pretrained=True).to(device)
    opt = torch.optim.AdamW([{"params": net.trunk.parameters(), "lr": args.lr},
                             {"params": net.head.parameters(), "lr": args.head_lr}], weight_decay=0.05)
    bce = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        net.train(); t0, run, seen = time.time(), 0.0, 0
        for step, (x, t, y, lab) in enumerate(tdl):
            if args.max_steps and step >= args.max_steps:
                break
            x, t, y = x.to(device), t.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logit, pl = net(x)
            loss = bce(pl.float(), t) + bce(logit.float(), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run += loss.item() * len(y); seen += len(y)
        m = evaluate(net, vdl, device)
        score = m.get("img_auroc_tampered_vs_real", 0.0) + m.get("patch_auroc_tampered", 0.0)
        print(f"epoch {epoch}: loss={run/max(seen,1):.4f} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "val": m}, args.out)
    print(f"best score {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()

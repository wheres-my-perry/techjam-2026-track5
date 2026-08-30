"""Fine-tune PE-Core-L14-336 (ViT-L/14) on manifest data. Kill-resumable.

    python -m src.approaches.pe_ft.train --train data/manifests/canon2_train.csv \
        --val data/manifests/canon2_val.csv --epochs 4 --augment \
        --crop-min 112 --crop-max 168 --batch 48 --out outputs/pe_ft/canon2.pt

Mirrors resnet_ft/train.py (approaches never import each other). Differences:
crop sides snap to multiples of 14 (patch size); bf16 autocast; the trunk
gets a small lr (1e-5) and the fresh head a large one (1e-3); grad clip 1.0.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter
from torch.utils.data import DataLoader, Dataset

from src.crops import clamp_size, random_crop, sample_size
from src.data import load_image, load_manifest
from src.metrics import auroc
from src.transforms import random_train_transform, style_aug, hard_train_transform, _stack as stack_transform
from .model import build_net, pick_device, to_tensor

CROP_MIN, CROP_MAX, CROP_STEP = 112, 168, 14


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, augment=False, crop=224, blur_boost=False, style=False, hard_aug=0.0, raw=False, stack_aug=0.0):
        self.samples = load_manifest(manifest_csv)
        self.augment = augment
        self.crop = crop
        self.blur_boost = blur_boost  # extra low-pass aug: forces blur-surviving cues
        self.style = style            # label-neutral style randomisation (both classes)
        self.hard_aug = hard_aug      # prob of ONE extreme corruption instead of the mild chain (both classes)
        self.raw = raw                # consistency mode: return the un-augmented image; views are made in the collate
        self.stack_aug = stack_aug    # prob of a random 2-or-3 transform STACK from the brief's grid (both classes; Thinh 2026-08-30)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s.path)
        if self.style:
            img = style_aug(img, random.Random())
        if self.raw:
            return img, float(s.label)
        if self.stack_aug and random.random() < self.stack_aug:
            img = stack_transform(img, random.choice((2, 3)), random.Random())
        elif self.hard_aug and random.random() < self.hard_aug:
            img = hard_train_transform(img, random.Random())
        elif self.augment:
            img = random_train_transform(img, random.Random())
        if self.blur_boost:
            rng = random.Random()
            p = rng.random()
            if p < 0.35:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))
            elif p < 0.60:
                w, h = img.size
                f = rng.uniform(0.25, 0.6)
                img = img.resize((max(8, int(w * f)), max(8, int(h * f))))
                img = img.resize((w, h))
        # NO crop here: a batch must stack into one tensor, so the crop size
        # is drawn once per batch in the collate fn (see make_collate).
        return img, float(s.label)


def make_collate(cmin, cmax, fixed=None):
    """Crop a whole batch to ONE randomly drawn size, then stack.

    Random SIZE (not just position) is Thinh's rule: the model never gets a
    constant input size to key on, and it sees the scale variety that the
    contest's resize transforms produce. Inference sweeps the same range via
    src.crops.size_ladder, so train and test crop the same way.
    """
    def collate(batch):
        rng = random.Random()
        size = fixed if fixed is not None else sample_size(rng, cmin, cmax, CROP_STEP)
        # one size for the batch: clamp to the smallest image so nothing upscales
        c = min(clamp_size(size, *im.size) for im, _ in batch)
        c = max(CROP_STEP, (c // CROP_STEP) * CROP_STEP)
        imgs = [random_crop(im, c, rng) for im, _ in batch]
        ys = torch.tensor([y for _, y in batch], dtype=torch.float32)
        return to_tensor(imgs), ys
    return collate


def make_consist_collate(cmin, cmax, k, hard_aug):
    """Consistency training (Thinh, 2026-08-30): ONE crop per image, K independently corrupted
    views of that crop (no geometry change, so the views show the same content and their
    embeddings should agree). Returns x of shape (K*B, C, H, W) ordered [view0 batch, view1 batch,
    ...] and y repeated K times; the trainer pairs view v of image i with view v' of image i."""
    def view_aug(im, rng):
        if hard_aug and rng.random() < hard_aug:
            return hard_train_transform(im, rng)
        return random_train_transform(im, rng, geometry=False)

    def collate(batch):
        rng = random.Random()
        size = sample_size(rng, cmin, cmax, CROP_STEP)
        c = min(clamp_size(size, *im.size) for im, _ in batch)
        c = max(CROP_STEP, (c // CROP_STEP) * CROP_STEP)
        crops = [random_crop(im, c, rng) for im, _ in batch]
        views = [[view_aug(cr, rng) for cr in crops] for _ in range(k)]
        ys = torch.tensor([y for _, y in batch], dtype=torch.float32)
        return to_tensor([im for v in views for im in v]), ys.repeat(k)
    return collate


def embedding_loss(e, kind, k, tau=0.1):
    """Agreement loss between the K views of each image. e: (K*B, D), view-major order."""
    z = torch.nn.functional.normalize(e.float(), dim=1)
    b = z.shape[0] // k
    zv = z.view(k, b, -1)
    if kind == "cos":  # 1 - cosine over all view pairs of the same image
        tot, n = 0.0, 0
        for i in range(k):
            for j in range(i + 1, k):
                tot = tot + (1 - (zv[i] * zv[j]).sum(1)).mean(); n += 1
        return tot / max(n, 1)
    if kind == "nce":  # NT-Xent: positives = other views of the same image, negatives = everything else
        sim = z @ z.t() / tau
        sim.fill_diagonal_(float("-inf"))
        img_id = torch.arange(b, device=z.device).repeat(k)
        pos = (img_id[:, None] == img_id[None, :]) & ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return -(logp.masked_fill(~pos, 0.0)).sum(1).div(pos.sum(1)).mean()
    raise ValueError(kind)


@torch.no_grad()
def evaluate_auroc(net, loader, device):
    net.eval()
    ys, ss = [], []
    for x, y in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = net(x.to(device)).squeeze(1).float()
        ss.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return auroc(ys, ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-5, help="trunk LR (pretrained ViT-L)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="fresh linear head LR")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="debug: stop each epoch after N steps")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--blur-boost", action="store_true",
                    help="extra blur/downscale aug on top of --augment (60%% of "
                         "samples get heavy low-pass; targets the measured "
                         "blur/resize weakness)")
    ap.add_argument("--crop", type=int, default=168,
                    help="fixed crop size; ignored when --crop-min/--crop-max "
                         "are given")
    ap.add_argument("--crop-min", type=int, default=None,
                    help=f"random-size crop range low end (e.g. {CROP_MIN})")
    ap.add_argument("--crop-max", type=int, default=None,
                    help=f"random-size crop range high end (e.g. {CROP_MAX})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/pe_ft/baseline.pt")
    ap.add_argument("--hard-aug", type=float, default=0.0,
                    help="probability that a sample gets ONE extreme corruption (blur 1.5-2.5, "
                         "resize 0.2-0.4, noise 0.07-0.12, jpeg 20-40) instead of the mild chain; "
                         "both classes alike (option B2, 2026-08-30)")
    ap.add_argument("--stack-aug", type=float, default=0.0,
                    help="probability that a sample gets a random 2-or-3 transform stack from the brief's grid "
                         "instead of the mild chain; both classes alike (Thinh: repost chains, 2026-08-30)")
    ap.add_argument("--consist", type=int, default=0,
                    help="K>0: augmentation-consistency training (Thinh): K corrupted views of the same "
                         "crop per image; loss = BCE(all views) + alpha * agreement loss")
    ap.add_argument("--consist-loss", default="cos", choices=["cos", "nce", "out"],
                    help="cos: 1-cosine between view embeddings; nce: NT-Xent contrastive on embeddings; "
                         "out: MSE between view output probabilities (control)")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the agreement loss")
    ap.add_argument("--tau", type=float, default=0.1, help="NT-Xent temperature")
    ap.add_argument("--style-aug", action="store_true",
                    help="label-neutral style randomisation on both classes (greyscale, saturation, tone, grain, vignette, flash)")
    ap.add_argument("--real-weight", type=float, default=1.0,
                    help="loss weight on REAL samples (label 0). >1 punishes false "
                         "positives (calling a real photo AI) harder. Thinh 2026-08-29: "
                         "per-crop verdicts must be conservative so an any-crop rule is safe.")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="seeded subsample of the train manifest (small-dataset trials)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    if device == "cuda":  # safe speed-ups (2026-08-30): TF32 matmul + cuDNN autotune; weights stay fp32, autocast bf16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    rand_size = args.crop_min is not None or args.crop_max is not None
    cmin = args.crop_min if args.crop_min is not None else args.crop
    cmax = args.crop_max if args.crop_max is not None else args.crop
    # val uses a FIXED mid size so val AUROC is comparable across epochs
    vfixed = (cmin + cmax) // 2 if rand_size else args.crop
    vfixed = max(CROP_STEP, (vfixed // CROP_STEP) * CROP_STEP)
    print(f"device={device} augment={args.augment} blur_boost={args.blur_boost} hard_aug={args.hard_aug} stack_aug={args.stack_aug} "
          f"crop={f'random {cmin}-{cmax}' if rand_size else args.crop} "
          f"(val fixed {vfixed})", flush=True)

    train_ds = ManifestDataset(args.train, args.augment, args.crop, blur_boost=args.blur_boost, style=args.style_aug,
                               hard_aug=args.hard_aug, raw=args.consist > 0, stack_aug=args.stack_aug)
    if args.limit_train and args.limit_train < len(train_ds.samples):
        rng = random.Random(args.seed)
        rng.shuffle(train_ds.samples)
        train_ds.samples = train_ds.samples[: args.limit_train]
        print(f"limit-train: {len(train_ds.samples)} rows", flush=True)
    train_collate = (make_consist_collate(cmin, cmax, args.consist, args.hard_aug) if args.consist > 0
                     else make_collate(cmin, cmax))
    train_dl = DataLoader(train_ds,
                          batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=train_collate)
    val_dl = DataLoader(ManifestDataset(args.val, False, args.crop),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=make_collate(cmin, cmax, fixed=vfixed))

    net = build_net(pretrained=True).to(device)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(
        [{"params": net.trunk.parameters(), "lr": args.lr},
         {"params": net.head.parameters(), "lr": args.head_lr}],
        weight_decay=0.05, fused=(device == "cuda"))
    amp = device == "cuda"
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(logits, y):
        # weight = real_weight on reals, 1 on fakes; mean over the batch
        w = torch.where(y < 0.5, torch.full_like(y, args.real_weight), torch.ones_like(y))
        return (bce(logits, y) * w).sum() / w.sum()
    print(f"loss: BCE with real_weight={args.real_weight}" + (
        f" + {args.alpha} * {args.consist_loss} agreement over {args.consist} views/image" if args.consist > 0 else ""), flush=True)

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
        t0, running, seen, run_agree = time.time(), 0.0, 0, 0.0
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                if args.consist > 0:
                    emb, logits = net.forward_feat(x)
                    logits = logits.squeeze(1)
                else:
                    logits = net(x).squeeze(1)
            loss = loss_fn(logits.float(), y)
            if args.consist > 0:
                if args.consist_loss == "out":
                    pv = torch.sigmoid(logits.float()).view(args.consist, -1)
                    agree = ((pv - pv.mean(0, keepdim=True)) ** 2).mean()
                else:
                    agree = embedding_loss(emb, args.consist_loss, args.consist, args.tau)
                loss = loss + args.alpha * agree
                run_agree += agree.item() * len(y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        val_auc = evaluate_auroc(net, val_dl, device)
        print(f"epoch {epoch}: loss={running/seen:.4f}" + (f" agree={run_agree/seen:.4f}" if args.consist > 0 else "") + f" val_auroc={val_auc:.4f} "
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

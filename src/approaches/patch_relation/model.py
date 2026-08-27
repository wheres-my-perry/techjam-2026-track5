"""Patch + relation head (approach 01, stage 2): attention over image patches.

Thinh's core insight, implemented: fakes are often locally plausible but
globally inconsistent (lighting, texture statistics, semantics that don't
agree across regions). A frozen fine-tuned ResNet-50 trunk describes each of
a 3x3 grid of native-resolution 224px patches; a small transformer lets the
patch descriptors ATTEND TO EACH OTHER before the verdict, so the model can
score cross-patch disagreement — not just per-patch evidence (which is what
crop-voting already captures).

Trunk is frozen (features come from resnet_ft's fine-tuned checkpoint); only
the ~1.6M-param relation head trains. Checkpoint is self-contained (trunk +
head weights both stored).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP = 224
GRID = 3
EMB = 2048  # resnet50 penultimate width
DIM = 256   # relation head width


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_tensor(images) -> torch.Tensor:
    arrs = [np.asarray(im, dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def grid_views(im, crop: int = CROP, grid: int = GRID):
    """3x3 grid of fixed-size crops at native resolution (same scheme as
    CropVoteModel in src/model.py; duplicated because approaches stay
    self-contained). Tiny images are upscaled so a crop exists. May return
    fewer than grid*grid views for small images (dedup of overlapping
    offsets) — pad_views() restores a fixed count."""
    c = crop
    w, h = im.size
    if min(w, h) < c:
        s = c / min(w, h)
        im = im.resize((max(c, round(w * s)), max(c, round(h * s))))
        w, h = im.size
    xs = sorted({round(t * (w - c) / max(1, grid - 1)) for t in range(grid)})
    ys = sorted({round(t * (h - c) / max(1, grid - 1)) for t in range(grid)})
    return [im.crop((x, y, x + c, y + c)) for y in ys for x in xs]


def pad_views(vs, n: int = GRID * GRID):
    """Small images yield <9 distinct crops; repeat cyclically to a fixed n so
    train and inference always see the same token count."""
    return (vs * n)[:n] if len(vs) < n else vs[:n]


def build_trunk() -> nn.Module:
    """ResNet-50 feature extractor (2048-d), fc stripped."""
    from torchvision.models import resnet50
    net = resnet50(weights=None)
    net.fc = nn.Identity()
    return net


def load_trunk_from_resnet_ft(ckpt_path: str) -> nn.Module:
    """Build trunk from a resnet_ft training checkpoint (its state_dict has a
    1-out fc; we load into the matching architecture then strip the fc)."""
    from torchvision.models import resnet50
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net.load_state_dict(ckpt["state_dict"])
    net.fc = nn.Identity()
    return net


class RelationHead(nn.Module):
    """Patch embeddings (B, P, 2048) -> logit (B,). P may vary per batch."""

    def __init__(self, n_pos: int = GRID * GRID, dim: int = DIM):
        super().__init__()
        self.proj = nn.Linear(EMB, dim)
        self.pos = nn.Parameter(torch.zeros(n_pos, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=2 * dim, dropout=0.1,
            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, P, EMB)
        p = x.shape[1]
        z = self.proj(x) + self.pos[:p].unsqueeze(0)
        z = self.enc(z)
        return self.head(z.mean(dim=1)).squeeze(1)


class PatchRelationModel(BaseModel):
    name = "patch_relation"
    CROP_BATCH = 96  # trunk forward chunk (crops, not images)

    def __init__(self, weights_path: str = "outputs/patch_relation/baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.trunk = build_trunk()
        self.trunk.load_state_dict(ckpt["trunk"])
        self.trunk.to(self.device).eval()
        self.rel = RelationHead()
        self.rel.load_state_dict(ckpt["head"])
        self.rel.to(self.device).eval()

    @torch.no_grad()
    def _embed_views(self, views) -> torch.Tensor:
        outs = []
        for i in range(0, len(views), self.CROP_BATCH):
            x = to_tensor(views[i:i + self.CROP_BATCH]).to(self.device)
            outs.append(self.trunk(x).float())
        return torch.cat(outs)  # (n_views, EMB)

    @torch.no_grad()
    def predict(self, images):
        P = GRID * GRID
        views = []
        for im in images:
            views.extend(pad_views(grid_views(im)))
        emb = self._embed_views(views)              # (len(images)*P, EMB)
        stack = emb.view(len(images), P, EMB)
        scores = np.zeros(len(images), dtype=np.float32)
        for i in range(0, len(images), 64):
            logits = self.rel(stack[i:i + 64].to(self.device))
            scores[i:i + 64] = torch.sigmoid(logits).float().cpu().numpy()
        return scores

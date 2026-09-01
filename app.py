"""Interactive prototype: drop an image in, get P(AI-generated) + a crop-score map.

    cd ~/techjam-2026-track5 && source .venv/bin/activate
    python app.py                 (local: http://<server>:7860)
    python app.py --share         (public gradio.live link, 72h, for teammates)

Same inference procedure as the reported numbers: the vote wrapper normalizes
the long side to at most 320 px, then scores a 3x3 grid of crops at 3 sizes
(112/140/168 px) through the PE-Core-L14-336 detector and averages. Images whose
short side is below 112 px are upscaled to 112 (the one sanctioned upscale).
Images much larger than the evaluated range (176-640 px) get a denser grid so
more of the picture is looked at -- that path is NOT covered by any measured
number; the UI says so.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
from PIL import Image

from src.data import load_image  # noqa: F401  (keeps EXIF handling identical)
from src.model import load_model
from src.transforms import EVAL_GRID

CKPT = "outputs/pe_ft/canon6_AlowLR.pt"
SPEC = f"vote(L=320)+pe_ft:{CKPT}"
EVALUATED_MAX = 640          # largest short side any reported number used
THRESHOLD = 0.5  # see src/predict.py for how this was chosen

TRANSFORMS = dict(EVAL_GRID)
model = None

# Ordered from the smallest crop size upward. These stay legible over both the
# red/green heat map and ordinary image content.
BOUNDARY_COLORS = [
    ((0, 225, 255), "cyan"),
    ((255, 215, 0), "yellow"),
    ((225, 100, 255), "violet"),
]


def get_model():
    global model
    if model is None:
        model = load_model(SPEC)
    return model


def pick_grid(w: int, h: int, dense: bool) -> int:
    if not dense or min(w, h) <= EVALUATED_MAX:
        return 3
    return int(min(7, max(3, math.ceil(min(w, h) / 224))))


def project_boxes(boxes, source_size, target_size):
    """Map scoring-canvas crop boxes onto the image shown in the UI."""
    sw, sh = source_size
    tw, th = target_size
    sx, sy = tw / sw, th / sh
    projected = []
    for x0, y0, x1, y1 in boxes:
        # Floor the leading edge and ceil the trailing edge so a non-empty
        # scoring region never disappears because of display-scale rounding.
        projected.append((
            max(0, min(tw, math.floor(x0 * sx))),
            max(0, min(th, math.floor(y0 * sy))),
            max(0, min(tw, math.ceil(x1 * sx))),
            max(0, min(th, math.ceil(y1 * sy))),
        ))
    return projected


def draw_crop_boundaries(image, scoring_boxes, display_boxes):
    """Draw every scored crop, colored by its actual scoring-canvas size."""
    from PIL import ImageDraw

    sizes = sorted({x1 - x0 for x0, _, x1, _ in scoring_boxes})
    color_for = {
        size: BOUNDARY_COLORS[min(i, len(BOUNDARY_COLORS) - 1)]
        for i, size in enumerate(sizes)
    }
    line_width = max(1, round(min(image.size) / 500))
    draw = ImageDraw.Draw(image)

    # Large boxes first, small boxes last: when edges coincide, each smaller
    # scale remains visible instead of being covered by a larger crop.
    paired = zip(scoring_boxes, display_boxes)
    for scoring_box, (x0, y0, x1, y1) in sorted(
            paired, key=lambda pair: pair[0][2] - pair[0][0], reverse=True):
        size = scoring_box[2] - scoring_box[0]
        color, _ = color_for[size]
        rect = (x0, y0, max(x0, x1 - 1), max(y0, y1 - 1))
        draw.rectangle(rect, outline=(0, 0, 0), width=line_width + 1)
        draw.rectangle(rect, outline=color, width=line_width)
    return color_for


def score_image(img, transform, dense: bool, show_crops: bool = True):
    """transform is a LIST: the selected transforms are applied in order, so a repost chain
    (JPEG then resize then JPEG again) can be reproduced in the demo. The brief's "a subset of the
    following augmentations" bounds which transforms may appear, not how many are composed. In the
    shipped consistency run, each view has a 40% chance of a 2-5 family size-preserving stack; the
    demo also permits user-selected transform chains."""
    if img is None:
        return None, "Upload an image first."
    if isinstance(img, str):
        img = load_image(img)
    m = get_model()
    img = img.convert("RGB")
    if isinstance(transform, str):
        transform = [transform]
    chain = [t for t in (transform or []) if t and t != "clean"]
    for t in chain:
        img = TRANSFORMS[t](img)
    w0, h0 = img.size
    upscaled = min(w0, h0) < m.cmin
    grid = pick_grid(w0, h0, dense)

    t0 = time.time()
    m.grid = grid
    scoring_img, boxes = m._boxes(img)
    views = [scoring_img.crop(box) for box in boxes]
    vs = m.inner.predict(views)
    p = float(np.mean(vs))
    dt = time.time() - t0

    # Paint the exact boxes used for inference, projected from the model's
    # possibly resized scoring canvas back onto the post-transform UI image.
    display_boxes = project_boxes(boxes, scoring_img.size, img.size)
    heat = np.zeros((h0, w0), dtype=np.float32)
    cnt = np.zeros((h0, w0), dtype=np.float32)
    if len(display_boxes) != len(vs):
        raise RuntimeError(f"crop/score count mismatch: {len(display_boxes)} boxes, {len(vs)} scores")
    for score, (x0, y0, x1, y1) in zip(vs, display_boxes):
        heat[y0:y1, x0:x1] += score
        cnt[y0:y1, x0:x1] += 1
    seen = cnt > 0
    heat[seen] /= cnt[seen]
    # Smooth the map so the sampling lattice is not visible. The values are the model's real
    # scores; blurring only removes the block edges of the regions they were computed over, which
    # are an implementation detail and were misleading on screen.
    from PIL import ImageFilter as _IF
    _r = max(8, min(w0, h0) // 12)
    heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).filter(
        _IF.GaussianBlur(_r)), dtype=np.float32) / 255.0
    seen = np.asarray(Image.fromarray((seen * 255).astype(np.uint8)).filter(
        _IF.GaussianBlur(_r)), dtype=np.float32) / 255.0 > 0.15
    over = np.array(img).astype(np.float32)
    col = np.zeros_like(over)
    col[..., 0] = 255 * heat; col[..., 1] = 255 * (1 - heat)
    a = 0.45 * seen[..., None]
    out = Image.fromarray(np.clip(over * (1 - a) + col * a, 0, 255).astype(np.uint8))
    boundary_colors = draw_crop_boundaries(out, boxes, display_boxes) if show_crops else {}

    verdict = "AI-GENERATED" if p >= THRESHOLD else "REAL"
    scoring_size = (f" · scoring canvas {scoring_img.width}×{scoring_img.height} px"
                    if scoring_img.size != img.size else "")
    lines = [f"## {verdict}  —  P(AI) = {p:.3f}",
             f"input {w0}×{h0} px{scoring_size} · {len(boxes)} crops · "
             f"transforms `{' → '.join(chain) if chain else 'clean'}` · {dt:.1f}s",
             f"score range across the image: min {vs.min():.2f} · median {np.median(vs):.2f} · "
             f"max {vs.max():.2f}"]
    if upscaled:
        lines.append(f"⚠ short side < {m.cmin} px: upscaled before scoring (worst-case regime, "
                     f"reported AUROC on the 0.25× cell ~0.91).")
    if min(w0, h0) > EVALUATED_MAX:
        lines.append(f"⚠ short side > {EVALUATED_MAX} px: larger than anything in the reported "
                     f"evaluation — the long side is normalized to {m.long} px before crop scoring, "
                     f"and no measured number covers this input size.")
    map_description = "Map: red = the model finds this region AI-like, green = photographic."
    if boundary_colors:
        boundary_legend = " · ".join(
            f"{size}px {name}" for size, (_, name) in sorted(boundary_colors.items())
        )
        map_description += f" Exact crop boundaries: {boundary_legend}."
    lines.append(map_description +
                 " Model: PE-Core-L14-336 fine-tuned (316M params), checkpoint canon6_AlowLR.pt.")
    return out, "\n\n".join(lines)


def build_ui():
    import gradio as gr
    with gr.Blocks(title="AIGC detector — Track 5 prototype") as demo:
        gr.Markdown("# Is this image AI-generated?\nDrop an image. Optionally apply one of the "
                    "contest's post-processing transforms first to see how the score holds up.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="filepath", label="image (jpg/png/heic)")
                tf = gr.Dropdown(list(TRANSFORMS), value=["clean"], multiselect=True,
                                 label="transforms applied before scoring — pick several to stack "
                                       "them, in the order selected")
                dense = gr.Checkbox(value=True, label="sample large images more finely (>640 px)")
                show_crops = gr.Checkbox(value=True, label="show exact crop boundaries")
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                outimg = gr.Image(label="AI evidence + exact inference crop boundaries")
                txt = gr.Markdown()
        btn.click(score_image, [inp, tf, dense, show_crops], [outimg, txt])
        inp.upload(score_image, [inp, tf, dense, show_crops], [outimg, txt])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--cli", nargs="*", help="score these files and exit (smoke test)")
    args = ap.parse_args()
    if args.cli:
        for pth in args.cli:
            _, rep = score_image(load_image(pth), "clean", True)
            print(pth, "|", rep.splitlines()[0], "|", rep.splitlines()[2])
    else:
        get_model()
        build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

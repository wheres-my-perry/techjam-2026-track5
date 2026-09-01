"""Interactive prototype: drop an image in, get P(AI-generated) + a crop-score map.

    cd ~/techjam-2026-track5 && source .venv/bin/activate
    python app.py                 (local: http://<server>:7860)
    python app.py --share         (public gradio.live link, 72h, for teammates)

Same inference procedure as the reported numbers: the vote wrapper scores a
3x3 grid of native-resolution crops at 3 sizes (112/140/168 px) through the
PE-Core-L14-336 detector and averages. Images whose short side is below 112 px
are upscaled to 112 (the one sanctioned upscale). Images much larger than the
evaluated range (176-640 px) get a denser grid so more of the picture is looked
at -- that path is NOT covered by any measured number; the UI says so.
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

from src.crops import grid_boxes, size_ladder
from src.data import load_image  # noqa: F401  (keeps EXIF handling identical)
from src.model import load_model
from src.transforms import EVAL_GRID

CKPT = "outputs/pe_ft/canon6_AlowLR.pt"
SPEC = f"vote(L=320)+pe_ft:{CKPT}"
EVALUATED_MAX = 640          # largest short side any reported number used
THRESHOLD = 0.5  # see src/predict.py for how this was chosen

TRANSFORMS = dict(EVAL_GRID)
model = None


def get_model():
    global model
    if model is None:
        model = load_model(SPEC)
    return model


def pick_grid(w: int, h: int, dense: bool) -> int:
    if not dense or min(w, h) <= EVALUATED_MAX:
        return 3
    return int(min(7, max(3, math.ceil(min(w, h) / 224))))


def score_image(img, transform, dense: bool):
    """transform is a LIST: the selected transforms are applied in order, so a repost chain
    (JPEG then resize then JPEG again) can be reproduced in the demo. The brief's "a subset of the
    following augmentations" bounds which transforms may appear, not how many are composed, and the
    model is trained on stacks of 2-6 -- so the demo has to be able to stack them too."""
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
    views = m._views(img)          # identical crops to evaluation
    vs = m.inner.predict(views)
    p = float(np.mean(vs))
    dt = time.time() - t0

    # per-crop map: paint each crop box with its own score (red = AI, green = real)
    wv, hv = (img.size if not upscaled else
              (max(m.cmin, round(w0 * m.cmin / min(w0, h0))),
               max(m.cmin, round(h0 * m.cmin / min(w0, h0)))))
    heat = np.zeros((hv, wv), dtype=np.float32)
    cnt = np.zeros((hv, wv), dtype=np.float32)
    i = 0
    for c in size_ladder(m.cmin, m.cmax, m.n_sizes, m.step):
        for (x0, y0, x1, y1) in grid_boxes(wv, hv, c, grid, m.step):
            heat[y0:y1, x0:x1] += vs[i]; cnt[y0:y1, x0:x1] += 1; i += 1
    seen = cnt > 0
    heat[seen] /= cnt[seen]
    base = img.resize((wv, hv)) if upscaled else img
    over = np.array(base).astype(np.float32)
    col = np.zeros_like(over)
    col[..., 0] = 255 * heat; col[..., 1] = 255 * (1 - heat)
    a = 0.45 * seen[..., None]
    out = Image.fromarray(np.clip(over * (1 - a) + col * a, 0, 255).astype(np.uint8))
    # No box overlay: the drawn grid used the display grid size, not the number of regions actually
    # scored, so it showed (for example) a 5x5 lattice over a differently-sampled score map. The
    # smooth heat map below is derived from the real scores and is the honest visualisation.

    verdict = "AI-GENERATED" if p >= THRESHOLD else "REAL"
    lines = [f"## {verdict}  —  P(AI) = {p:.3f}",
             f"input {w0}×{h0} px · transforms `{' → '.join(chain) if chain else 'clean'}` · {dt:.1f}s",
             f"score range across the image: min {vs.min():.2f} · median {np.median(vs):.2f} · "
             f"max {vs.max():.2f}"]
    if upscaled:
        lines.append(f"⚠ short side < {m.cmin} px: upscaled before scoring (worst-case regime, "
                     f"reported AUROC on the 0.25× cell ~0.91).")
    if min(w0, h0) > EVALUATED_MAX:
        lines.append(f"⚠ short side > {EVALUATED_MAX} px: larger than anything in the reported "
                     f"evaluation — the image is still scored at native resolution, but no measured number "
                     f"covers this size.")
    lines.append("Map: red = the model finds this region AI-like, green = photographic. "
                 "Model: PE-Core-L14-336 fine-tuned (316M params), checkpoint canon6_AlowLR.pt.")
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
                btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                outimg = gr.Image(label="where the model sees AI evidence")
                txt = gr.Markdown()
        btn.click(score_image, [inp, tf, dense], [outimg, txt])
        inp.upload(score_image, [inp, tf, dense], [outimg, txt])
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

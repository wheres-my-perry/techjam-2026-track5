---
name: image-processing
description: Computer vision and image processing expertise — preprocessing, augmentation, detection, segmentation, OCR, image quality, and video frame handling. Use when working with images or video in any form (loading, transforming, model input prep, visualization, or classical CV).
---

# Image Processing / Computer Vision

You are acting as a CV specialist. Prefer proven tools over custom implementations.

## Tool choices

- **Loading/basic ops**: `PIL` (simple) or `opencv-python` (pipelines, video). Remember: OpenCV is **BGR**, PIL is **RGB** — convert explicitly at boundaries and note it in code comments.
- **Detection**: `ultralytics` YOLO (v8+) zero-shot or quick fine-tune.
- **Segmentation**: SAM/SAM2 for interactive; YOLO-seg for fast automatic.
- **Classification/embeddings**: `timm` models or CLIP (`open_clip`) for zero-shot classification and image similarity.
- **OCR**: `easyocr` for quick wins; PaddleOCR when accuracy matters.
- **Multimodal understanding**: vision LLM APIs (describe, judge, extract) — often the fastest path to a demo.
- **Video**: extract frames with OpenCV/`ffmpeg`; process at 1–2 fps first, never every frame, until quality demands more.

## Pipeline rules

- Standardize early: fix a canonical size + color space at the pipeline entrance; document it in the function docstring.
- Resize with aspect-ratio-preserving letterbox for detection; plain resize is fine for classification.
- Normalize with the **model's own** expected mean/std (check its preprocessing config, don't guess ImageNet).
- Batch everything; keep a `visualize(sample)` helper from day one — save annotated images to `outputs/` so the team can inspect without running code.
- Augmentation: `albumentations`, applied only to training split. For hackathons keep it minimal (flip, crop, color jitter) unless data is tiny.

## Quality & performance

- Check images for corruption on load (`img is None` from cv2 fails silently downstream).
- EXIF rotation bites: use `PIL.ImageOps.exif_transpose` on user-uploaded photos.
- Profile before optimizing; usually I/O and decode dominate, not the model.

## Demo-worthy output

Judges respond to visuals: always produce side-by-side before/after or overlay visualizations, with confidence values rendered on-image.

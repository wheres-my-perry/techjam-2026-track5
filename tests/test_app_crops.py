"""Regression tests for inference crops and their UI projection."""

import numpy as np
from PIL import Image

import app
from src.model import BaseModel, CropVoteModel


class ConstantCropModel(BaseModel):
    name = "constant"
    CROP_MIN, CROP_MAX, CROP_STEP = 112, 168, 14

    def predict(self, images):
        return np.full(len(images), 0.25, dtype=np.float32)


def test_tiny_input_boxes_use_upscaled_dimensions():
    model = CropVoteModel(ConstantCropModel(), long=320)
    scoring_img, boxes = model._boxes(Image.new("RGB", (50, 100)))

    assert scoring_img.size == (112, 224)
    assert boxes
    assert all(0 <= x0 < x1 <= 112 and 0 <= y0 < y1 <= 224
               for x0, y0, x1, y1 in boxes)
    assert {x1 - x0 for x0, _, x1, _ in boxes} == {112}


def test_project_boxes_maps_scoring_canvas_to_ui_image():
    boxes = [(0, 0, 112, 112), (208, 128, 320, 240)]
    assert app.project_boxes(boxes, (320, 240), (640, 480)) == [
        (0, 0, 224, 224),
        (416, 256, 640, 480),
    ]


def test_draw_crop_boundaries_uses_size_colors():
    image = Image.new("RGB", (100, 100))
    scoring_boxes = [(10, 10, 30, 30), (10, 10, 50, 50)]
    display_boxes = list(scoring_boxes)

    colors = app.draw_crop_boundaries(image, scoring_boxes, display_boxes)

    assert colors == {
        20: ((0, 225, 255), "cyan"),
        40: ((255, 215, 0), "yellow"),
    }
    # Both boxes start here; the smaller box is drawn last and stays visible.
    assert image.getpixel((10, 10)) == (0, 225, 255)


def test_score_image_uses_actual_boxes_after_long_side_resize(monkeypatch):
    # 1000x437 becomes exactly 320x140. Some crop sizes then collapse onto
    # fewer grid positions, which used to desynchronise the heat map and scores.
    monkeypatch.setattr(app, "model", CropVoteModel(ConstantCropModel(), long=320))
    output, report = app.score_image(Image.new("RGB", (1000, 437)), "clean", False)

    assert output.size == (1000, 437)
    assert "scoring canvas 320×140 px" in report
    assert "15 crops" in report
    assert "112px cyan · 140px yellow" in report


def test_score_image_can_hide_crop_boundaries(monkeypatch):
    monkeypatch.setattr(app, "model", CropVoteModel(ConstantCropModel(), long=320))
    image = Image.new("RGB", (224, 224))

    with_boundaries, shown_report = app.score_image(image, "clean", False, True)
    without_boundaries, hidden_report = app.score_image(image, "clean", False, False)

    assert with_boundaries.getpixel((0, 0)) == (0, 225, 255)
    assert without_boundaries.getpixel((0, 0)) != (0, 225, 255)
    assert "Exact crop boundaries:" in shown_report
    assert "Exact crop boundaries:" not in hidden_report

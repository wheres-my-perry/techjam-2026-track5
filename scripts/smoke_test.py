"""End-to-end smoke test of the harness with synthetic data. No downloads needed.

Creates 40 synthetic images (dark = real, bright = fake), a manifest, then runs
src.evaluate with the 'brightness' toy model and src.predict on the directory.

Run from repo root:  python scripts/smoke_test.py
"""

import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "outputs", "smoke")


def make_data():
    img_dir = os.path.join(TMP, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        label = i % 2  # 1 = fake = bright, 0 = real = dark
        base = 170 if label else 80
        arr = np.clip(rng.normal(base, 30, (96, 96, 3)), 0, 255).astype(np.uint8)
        p = os.path.join(img_dir, f"img_{i:03d}.png")
        Image.fromarray(arr).save(p)
        rows.append({"path": p, "label": label, "generator": "toy", "source": "toy"})
    man = os.path.join(TMP, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source"])
        w.writeheader()
        w.writerows(rows)
    return img_dir, man


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    img_dir, man = make_data()
    run([sys.executable, "-m", "src.evaluate", "--manifest", man,
         "--model", "brightness", "--out", os.path.join(TMP, "eval")])
    run([sys.executable, "-m", "src.predict", "--input", img_dir,
         "--output", os.path.join(TMP, "preds.json"), "--model", "brightness"])
    print("\nSmoke test OK.")


if __name__ == "__main__":
    main()

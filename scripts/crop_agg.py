"""Compare crop-aggregation rules OFFLINE on a per-crop dump (scripts/crop_dump.py).

  python -m scripts.crop_agg outputs/random_gen/dump_grid.npz [more.npz ...] [--fa 0.01 0.05]

For each dump prints coverage evenness (how many crops cover a pixel: min / max / mean over
pixels, averaged over images) and, per rule, pooled AUROC and the share of fakes caught when the
cut-off is set so that 1% (5%) of the real photos are wrongly flagged.

Rules (image score from its crop scores):
  mean      plain average of all crops (shipped)
  size_mean average of per-size averages (each crop size counts equally, not each crop)
  area      crops weighted by their area (Thinh's suggestion)
  pixel     every PIXEL counts equally: a pixel's score = mean of the crops covering it,
            image score = mean over pixels (undoes uneven coverage)
  median    median of crops
  trim10    mean after dropping the top and bottom 10% of crops
  top3      mean of the 3 highest crops
"""
from __future__ import annotations
import argparse, numpy as np
from sklearn.metrics import roc_auc_score


def per_image(d):
    """Yield (image index, scores, boxes) for one dump, in image order."""
    order = np.argsort(d["img"], kind="stable")
    img = d["img"][order]; sc = d["score"][order]
    bx = np.stack([d["x0"], d["y0"], d["x1"], d["y1"]], 1)[order].astype(int)
    starts = np.r_[0, np.flatnonzero(np.diff(img)) + 1, len(img)]
    for a, b in zip(starts[:-1], starts[1:]):
        yield img[a], sc[a:b], bx[a:b]


def pixel_score(sc, bx, w, h):
    tot = np.zeros((h, w), np.float64); cnt = np.zeros((h, w), np.int32)
    for s, (x0, y0, x1, y1) in zip(sc, bx):
        tot[y0:y1, x0:x1] += s; cnt[y0:y1, x0:x1] += 1
    m = cnt > 0
    return float((tot[m] / cnt[m]).mean()), cnt


def rules(sc, bx, w, h):
    area = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])).astype(float)
    sizes = bx[:, 2] - bx[:, 0]
    out = {"mean": sc.mean(),
           "size_mean": np.mean([sc[sizes == c].mean() for c in np.unique(sizes)]),
           "area": float((sc * area).sum() / area.sum()),
           "median": float(np.median(sc)),
           "top3": float(np.sort(sc)[::-1][:3].mean())}
    k = int(0.1 * len(sc))
    out["trim10"] = float(np.sort(sc)[k:len(sc) - k].mean()) if len(sc) - 2 * k > 0 else sc.mean()
    px, cnt = pixel_score(sc, bx, w, h)
    out["pixel"] = px
    return out, cnt


def evaluate(scores, labels, fas):
    r = scores[labels == 0]; f = scores[labels == 1]
    res = [roc_auc_score(labels, scores)]
    for fa in fas:
        t = np.quantile(r, 1 - fa); res.append((f >= t).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fa", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--save", default="", help="write per-image rule scores to CSV (last dump only)")
    a = ap.parse_args()
    for path in a.dumps:
        d = np.load(path, allow_pickle=False)
        labels = d["label"].astype(int); n = len(labels)
        w, h = d["w"], d["h"]
        names = ["mean", "size_mean", "area", "pixel", "median", "trim10", "top3"]
        S = {k: np.zeros(n) for k in names}
        cov_min, cov_max, cov_mean, ncrops = [], [], [], []
        for i, sc, bx in per_image(d):
            out, cnt = rules(sc, bx, int(w[i]), int(h[i]))
            for k in names: S[k][i] = out[k]
            cov_min.append(cnt.min()); cov_max.append(cnt.max()); cov_mean.append(cnt.mean()); ncrops.append(len(sc))
        print(f"\n== {path}  model {d['model']}")
        print(f"   {n} images ({(labels == 1).sum()} fakes / {(labels == 0).sum()} reals), crops per image "
              f"{np.mean(ncrops):.1f}; pixel coverage min {np.mean(cov_min):.1f} / max {np.mean(cov_max):.1f} / "
              f"mean {np.mean(cov_mean):.1f} (averaged over images); uncovered pixels: "
              f"{np.mean([c == 0 for c in cov_min]) * 100:.1f}% of images have any")
        head = "   rule       AUROC   " + "  ".join(f"caught@{fa:.0%}FA" for fa in a.fa)
        print(head)
        for k in names:
            res = evaluate(S[k], labels, a.fa)
            print(f"   {k:9s}  {res[0]:.4f}  " + "  ".join(f"{v * 100:11.1f}%" for v in res[1:]))
        if a.save:
            import csv
            with open(a.save, "w", newline="") as fh:
                wr = csv.writer(fh); wr.writerow(["set", "label"] + names)
                for i in range(n): wr.writerow([d["set"][i], labels[i]] + [f"{S[k][i]:.5f}" for k in names])


if __name__ == "__main__":
    main()

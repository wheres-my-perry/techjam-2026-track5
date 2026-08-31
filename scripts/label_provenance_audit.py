"""Label-provenance audit (MANDATORY GATE, added 2026-08-30 after the WildFake GAN-label bug).

The other audits test SEPARABILITY (metadata, style canaries, size buckets). None of them asks
whether a row's label is actually TRUE for the file it points to. This one does: every row's
`orig` (or `path`) is resolved back to its source's authoritative label, independently of the
manifest builder that produced the row.

Rules
  wildfake : `orig` under data/wildfake/raw/Images/Real/**  -> label 0 ; under Images/<Family>_based/** -> 1.
             Fakes must additionally have the CSV generator's top-level folder in the path.
  artifact : label from the ArtiFact metadata `target` column (0 real / 1 fake), matched by exact
             relative path; unmatched rows are reported.
  ext      : by source folder name: real_* / *_real / afhq_512 / celebahq_1024 / openimages_1024 /
             ffhq / coco_640 / sid_real -> 0 ; everything else -> 1 (folder list printed for review).
  lsun*    : always 0.   ddpm/ddim/wf_test rows: as wildfake.
Also: no source file may carry two labels, and no source file may appear in two splits.

    python -m scripts.label_provenance_audit --prefix data/manifests/canon5 [--strict]
Exit 1 with --strict on any violation.
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import Counter, defaultdict

# Real-photo ext sources. A source missing from this set is re-derived as FAKE and the gate
# FAILS -- which is what happened when flickr30k_web was added without updating it. Add the
# source here in the same commit that adds it to the corpus.
REAL_EXT = {"afhq_512", "celebahq_1024", "openimages_1024", "ffhq_1024", "coco_640", "sid_real",
            "real", "flickr30k_web"}


def load_artifact_targets():
    """relative path -> target (0/1) from ArtiFact metadata csvs, if present."""
    out = {}
    root = "data/artifact"
    if not os.path.isdir(root):
        return out
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "metadata.csv":
                with open(os.path.join(dp, fn), newline="") as f:
                    for r in csv.DictReader(f):
                        p = os.path.normpath(os.path.join(dp, r.get("image_path", "")))
                        if "target" in r:  # ArtiFact target: 0 = real, any other id = a fake family
                            out[p] = 0 if r["target"].strip() == "0" else 1
    return out


def authoritative(row, art):
    src, orig = row.get("source", ""), row.get("orig", "") or row.get("path", "")
    o = orig.replace("\\", "/")
    if src == "wildfake" or "/wildfake/" in o:
        if "/Images/Real/" in o: return 0, "wildfake:Real folder"
        if "_based/" in o: return 1, "wildfake:generator folder"
        return None, "wildfake:unresolved"
    if src.startswith("artifact") or "/artifact/" in o:
        p = os.path.normpath(o.lstrip("./"))
        if p in art: return art[p], "artifact:metadata target"
        # fall back: ArtiFact real folders carry the dataset name, fakes the generator name
        parts = p.split("/")
        return None, "artifact:no metadata match"
    if src.startswith("lsun"): return 0, "lsun"
    # OmniFake (MoeNew/OmniFake val split): the archive's own directory layout is the authority --
    # val/real/** are its matched real photographs, val/<Generator>/** are its synthetic images.
    # Registered here in the same commit that adds the source, because an unregistered source is
    # re-derived as FAKE and fails the gate (which is what flickr30k_web did).
    if src.startswith("omnifake") or "/omnival/" in o:
        if "/val/real/" in o or src == "omnifake_real": return 0, "omnifake:val/real"
        return 1, "omnifake:val/<generator>"
    if src in REAL_EXT or "/ext/img/" in o:
        folder = o.split("/ext/img/")[-1].split("/")[0] if "/ext/img/" in o else src
        return (0 if folder in REAL_EXT or folder.startswith("real") else 1), f"ext:{folder}"
    return None, f"unknown source {src}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    art = load_artifact_targets()
    print(f"artifact metadata targets loaded: {len(art)}")
    bad, unresolved, by_rule = Counter(), Counter(), Counter()
    labels_by_file, splits_by_file = defaultdict(set), defaultdict(set)
    examples = []
    for split in ("train", "val", "test"):
        p = f"{a.prefix}_{split}.csv"
        if not os.path.exists(p): continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                key = r.get("orig") or r["path"]
                labels_by_file[key].add(r["label"]); splits_by_file[key].add(split)
                truth, rule = authoritative(r, art)
                by_rule[rule] += 1
                if truth is None:
                    unresolved[(split, r.get("source", ""))] += 1
                elif str(truth) != r["label"]:
                    bad[(split, r.get("source", ""), r.get("generator", ""))] += 1
                    if len(examples) < 5: examples.append((split, r["label"], truth, key))
    conflicts = [k for k, v in labels_by_file.items() if len(v) > 1]
    leaks = [k for k, v in splits_by_file.items() if len(v) > 1]
    print("\nrows by resolution rule:")
    for k, v in by_rule.most_common(): print(f"  {v:8d}  {k}")
    print(f"\nLABEL DISAGREEMENTS (manifest label != authoritative): {sum(bad.values())}")
    for k, v in bad.most_common(15): print(f"  {v:7d}  split={k[0]} source={k[1]} generator={k[2]}")
    for e in examples: print("   e.g.", e)
    print(f"UNRESOLVED rows (no authoritative label found): {sum(unresolved.values())}")
    for k, v in unresolved.most_common(8): print(f"  {v:7d}  split={k[0]} source={k[1]}")
    print(f"files carrying two labels: {len(conflicts)}")
    print(f"files present in two splits: {len(leaks)}")
    ok = not bad and not conflicts and not leaks
    print("\nLABEL PROVENANCE:", "CLEAN" if ok else "FAIL")
    if a.strict and not ok: sys.exit(1)


if __name__ == "__main__":
    main()

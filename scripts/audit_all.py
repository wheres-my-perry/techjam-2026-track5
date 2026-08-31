"""Run EVERY gate on a manifest and print one verdict table.

Written 2026-08-31 after canon6 was nearly reported having skipped size_audit
entirely, and after the audits that DID run turned out to be blind to the thing
that mattered. Running the gates one at a time from memory is how a gate gets
skipped; this makes the full set one command.

Two failure modes this encodes, both learned the hard way:

  1. A SKIPPED GATE. CLAUDE.md binds every manifest to label_provenance +
     shortcut + size_audit. size_audit was skipped for canon6 because the other
     three had been run and it felt covered. Gates are a checklist, not a vibe.

  2. A GATE THAT CANNOT SEE THE PROBLEM. After canonicalization every image is
     176x176 PNG, so shortcut_audit and size_audit -- which both read the
     CANONICAL files -- see constant width/height/format and can only vary on
     file size. They will happily pass a set in which native size predicts the
     label perfectly. canon_unseen6 passed shortcut_audit at 0.617 while three
     of its five native-size buckets contained no fakes at all and reals ran to
     7712px against fakes capped at 1024. The shrink-to-320 factor leaves a
     physical trace, so "big => real" is learnable without looking at content
     (docs/LESSONS_FOR_TEAMMATES.md section 2).
     => native size is checked HERE, from the manifest's `long` column, because
        no audit that reads the canonical pixels can ever check it.

    python -m scripts.audit_all --prefix data/manifests/canon6          # train/val/test
    python -m scripts.audit_all --manifest data/manifests/canon_unseen6.csv --eval-set
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

ORDER = ["<=341", "342-512", "513-768", "769-1024", ">1024"]


def buck(v):
    v = int(v)
    if v <= 341: return "<=341"
    if v <= 512: return "342-512"
    if v <= 768: return "513-768"
    if v <= 1024: return "769-1024"
    return ">1024"


def run(cmd):
    p = subprocess.run([sys.executable, "-m"] + cmd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."})
    return p.stdout + p.stderr


def native_size(manifest, eval_set):
    """The check no canonical-pixel audit can do. Returns (verdict, lines)."""
    rows = list(csv.DictReader(open(manifest, newline="")))
    if not rows or "long" not in rows[0]:
        return "SKIP", ["  no `long` column — cannot check native size"]
    by = defaultdict(lambda: [0, 0])
    vals = defaultdict(list)
    for r in rows:
        b = buck(r["long"])
        by[b][int(r["label"])] += 1
        vals[r["label"]].append(int(r["long"]))
    lines, one_class, skewed = [], [], []
    for b in ORDER:
        if b not in by:
            continue
        nr, nf = by[b]
        if not nr or not nf:
            one_class.append(b)
            lines.append(f"  {b:10s} real {nr:7d} fake {nf:7d}   ONE CLASS ONLY")
            continue
        ratio = nr / nf
        bad = not (0.8 <= ratio <= 1.25)
        if bad:
            skewed.append(f"{b} {ratio:.2f}:1")
        lines.append(f"  {b:10s} real {nr:7d} fake {nf:7d}   {ratio:6.2f}:1"
                     + ("   SKEWED" if bad else ""))
    for lab, nm in (("0", "real"), ("1", "fake")):
        v = sorted(vals[lab])
        if v:
            lines.append(f"  {nm} native long: min {v[0]} median {v[len(v)//2]} max {v[-1]}")
    if one_class or skewed:
        if eval_set:
            lines.append("  => EVAL SET: report SIZE-MATCHED only "
                         "(python -m scripts.size_matched ...). A pooled number over "
                         "these buckets is partly a measurement of image size.")
            return "CAVEAT", lines
        lines.append("  => TRAINING data must be balanced in every bucket; fix the corpus.")
        return "FAIL", lines
    return "PASS", lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", help="data/manifests/canon6 -> _train/_val/_test")
    ap.add_argument("--manifest", help="a single manifest")
    ap.add_argument("--eval-set", action="store_true",
                    help="evaluation set: bucket imbalance is a reporting CAVEAT, not a failure")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    if not (a.prefix or a.manifest):
        raise SystemExit("need --prefix or --manifest")

    verdicts = []
    if a.prefix:
        out = run(["scripts.label_provenance_audit", "--prefix", a.prefix])
        ok = "LABEL PROVENANCE: CLEAN" in out
        verdicts.append(("label provenance", "PASS" if ok else "FAIL",
                         "labels re-derived from source" if ok else out.strip()[-200:]))
        out = run(["scripts.bucket_audit", "--prefix", a.prefix])
        verdicts.append(("bucket balance", "PASS" if "BUCKET AUDIT: CLEAN" in out else "FAIL", ""))
        targets = [f"{a.prefix}_train.csv"]
    else:
        targets = [a.manifest]

    for t in targets:
        out = run(["scripts.shortcut_audit", "--manifest", t])
        m = re.search(r"metadata-only AUROC:\s*([0-9.]+)", out)
        v = float(m.group(1)) if m else float("nan")
        verdicts.append((f"shortcut (metadata) {os.path.basename(t)}",
                         "PASS" if v <= 0.6 else "CAVEAT" if v <= 0.65 else "FAIL",
                         f"AUROC {v:.4f} — canonical files only: BLIND to native size"))

        out = run(["scripts.canary_audit", "--manifest", t, "--limit", str(a.limit)])
        m = re.search(r"WORST CANARY:\s*([0-9.]+)", out)
        v = float(m.group(1)) if m else float("nan")
        verdicts.append((f"canary (dumb pixels) {os.path.basename(t)}",
                         "PASS" if v <= 0.65 else "CAVEAT",
                         f"worst {v:.4f}" + ("" if v <= 0.65 else " — verify on the CHECKPOINT "
                                             "(scripts.style_check), not the manifest")))

        out = run(["scripts.size_audit", "--manifest", t, "--limit", str(a.limit)])
        sizes = set(re.findall(r"(\d+x\d+):", out))
        verdicts.append((f"size_audit {os.path.basename(t)}",
                         "PASS" if len(sizes) <= 1 else "CHECK",
                         f"{len(sizes)} distinct canonical size(s): {sorted(sizes)[:3]}"))

        vd, lines = native_size(t, a.eval_set)
        verdicts.append((f"NATIVE size {os.path.basename(t)}", vd,
                         "the one no canonical-pixel audit can do"))
        print(f"\nnative-size distribution — {t}")
        print("\n".join(lines))

    print(f"\n{'gate':46s} {'verdict':8s} note")
    print("-" * 110)
    worst = 0
    for name, vd, note in verdicts:
        worst = max(worst, {"PASS": 0, "SKIP": 0, "CHECK": 1, "CAVEAT": 1, "FAIL": 2}[vd])
        print(f"{name:46s} {vd:8s} {note}")
    print("\nAny CAVEAT must appear beside every number reported from this manifest.")
    sys.exit(2 if worst == 2 else 0)


if __name__ == "__main__":
    main()

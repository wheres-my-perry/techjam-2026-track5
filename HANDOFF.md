# HANDOFF — data & model state (2026-08-30, 22:10 server time)

Read in this order: this file → `docs/DATA_STATUS_2026-08-30.md` (complete counts/sources/flaws) →
`docs/DATA_AUDIT_2026-08-30.md` (every audit with its command and raw output) → `docs/FINDINGS.md` (log).

## The one thing to know
canon2/3/4 training data had a label bug: **every WildFake "GAN" row (stylegan, vqvae, biggan, stargan) was a
real AFHQ/FFHQ photo labelled fake** — 24.5 % of the claimed training fakes. Cause: `scripts/get_wildfake.py`
matched label CSVs to files by filename only (fixed). Found by the teammate's audit, verified in our manifests.
**canon4 is retired. Every GAN number ever quoted is void. canon5 is the corrected data.**

## canon5 (current data) — `data/manifests/canon5_{train,val,test}.csv` (server, git-ignored)
- train 296,092 (148,056 real / 148,036 fake), val 36,488 (18,245 / 18,243), test 191,759 (100,140 / 91,619, unbalanced by design)
- train/val: real == fake in every native-size bucket (≤341 / 342–512 / 513–768 / 769–1024 px)
- removed: all bogus GAN rows, duplicate rows, cross-split files, 78 blank canonical PNGs, val/test perceptual near-duplicates of train
- gates: label provenance CLEAN (0 disagreements / 0 dual labels / 0 cross-split) · bucket CLEAN · metadata-only AUROC 0.63 (mild)
  · **style canary 0.68 = FAIL** (line 0.65; the mislabelled reals had masked the true fake-vs-real style gap → check the
  trained checkpoint with greyscale / channel-swap scoring, which we do)
- sha256[:16]: train c9caf050036fd07c · val fb7b7ce2d31ca174 · test 92032a3927a0a4a3

## Benchmarks — flaws you must know before quoting a number
- **Judges' set** (DALL·E-3 vs COCO val2017): labels verified; metadata CLEAN; colour/style canaries fail (contest-data property);
  1,200-image eval subsample = 1,137 unique files; never trained on. Numbers stand.
- **Unseen-64 generators**: was 31 % duplicate rows → now `randtest_unique` (11,729 unique: 10,829 fakes, 900 reals never trained on).
  FLAW: metadata-separable (reals varied-size JPEG, fakes fixed-size PNG) → report SIZE-MATCHED only; ≤341 px and >1024 px buckets
  have no unseen reals and are not scorable yet. 12 of 64 sources are synthetic by inference (no dataset card). Per-source n is
  small for Rapidata sources (FLUX-2 Pro 7, Hunyuan 23, Halfmoon 31, Seedream 33, Ideogram 54) — always quote n.
- **Wild** (5 iPhone + 5 Gemini): 0 overlap with training.

## Numbers that stand (canon4 checkpoint, cut-off 0.15)
DALL·E benchmark 0.9999 clean / 0.996 mean over 14 corruptions AUROC; at 0.15: fakes 100 % / 99 % caught, reals flagged
2.9 % clean / 10.2 % mean / 26.7 % worst (degraded small reals are the weak side). Unseen (unique, size-matched, 46 sources):
0.9955 AUROC, ~94–95 % caught, 1.0 % flagged. Wild 10/10. DIV2K 2K reals 9 % flagged.

## In flight (Slurm, physical GPU 0)
job 158 nce consistency evals → job 180 cos evals → **job 178 canon5 clean retrain + full evals** (same recipe as canon4).
Compare candidates with `python -m scripts.model_card <names>` (each at its own 1 %-FA cut-off, never a shared raw threshold).

## How to verify any of this yourself
```
cd ~/techjam-2026-track5 && PYTHONPATH=. .venv/bin/python -m scripts.label_provenance_audit --prefix data/manifests/canon5 --strict
cd ~/techjam-2026-track5 && PYTHONPATH=. .venv/bin/python -m scripts.label_provenance_audit --prefix data/manifests/canon4
cd ~/techjam-2026-track5 && PYTHONPATH=. .venv/bin/python scripts/bucket_audit.py --prefix data/manifests/canon5 --strict
cd ~/techjam-2026-track5 && PYTHONPATH=. .venv/bin/python -m scripts.hash_analyze --csv outputs/audit/hash_audit.csv --maxd 2 --exclude-trivial
```
(first → CLEAN; second → 59,995 disagreements, FAIL — the gate reproduces the bug to the row)

## Binding rules from now on
No manifest is trained on or reported without: label_provenance_audit --strict, bucket_audit --strict, shortcut_audit,
canary_audit. Never match dataset label files to images by basename. Every reported number states the set, its n, and
the cut-off rule. All GPU work via sbatch.

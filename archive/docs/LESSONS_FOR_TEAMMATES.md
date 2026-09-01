# Lessons from 5 days of AIGC-detection experiments — read before touching data

Self-contained handoff for another team/agent (TikTok TechJam 2026 Track 5: real-vs-AI image
detection, robust to JPEG/blur/resize/noise/jitter/crop, models < 2B params). Everything below
was MEASURED on our runs (repo: github.com/wheres-my-perry/techjam-2026-track5; tools referenced
by path). Written 2026-08-31 after our GPU server died with all checkpoints on it.

## 0. The one rule that would have saved us two days
**Validate the benchmark model-free before believing any model number.** Every dataset we
touched had a shortcut that let a classifier "win" without looking at content. A perfect or
near-perfect score (>= 0.99 anywhere, or an AUROC that jumps when nothing about detection
changed) is an accusation, not a result: hunt the shortcut first, celebrate never.

Gate every manifest with dumb models BEFORE training:
- `scripts/shortcut_audit.py` — logistic regression on metadata only (width, height, aspect,
  file size, format). Must score ~0.5. >0.65 = FAIL, results unreportable.
- `scripts/size_audit.py` — per-class / per-generator size table (eyeball: disjoint = leak).
- `scripts/canary_audit.py` — deliberately dumb PIXEL models (colour histogram etc.) must
  score ~0.5; `scripts/content_audit.py` flags one-sided subjects (e.g. "church = always real").
- Run them with `--strict` inside job chains so nothing trains on a FAILed manifest.

## 1. State of every dataset we used (measured)

| dataset | what we found | fix / status |
|---|---|---|
| **WildFake (ModelScope hy2628982280/WildFake)** | ALL reals shipped as 200x200 thumbnails; GAN fakes (biggan/stargan/stylegan/vqvae) 200x200; diffusion fakes (ddim/ddpm) 256x256 -> size = label for diffusion. WORSE: the CSV->file matching by basename mislabelled every WildFake GAN row — those "fakes" were real AFHQ/FFHQ photos (24.5% of claimed training fakes). Its COCO slice is the official-benchmark real class (never train on it). | Match labels by full path + metadata, not basename. Size fixed by native random crop (below). Every pre-fix GAN number is void. |
| **WildFake DALL-E + COCO ("official" benchmark)** | reals 200x200 vs DALL-E 1024-1792 -> metadata-only AUROC ~1.0. Rebuilt with original-res COCO val2017 (375-640) = still size-gapped and it FAILS the colour canary (0.755): DALL-E palette != COCO palette. | Contest-provided; cannot be fixed, only caveated. Never quote it as your main number. |
| **ArtiFact (HF bitmind/ArtiFact, 2.5M imgs, 25 generators)** | Genuinely size-uniform 200x200 both classes + randomized JPEG quality (its authors defended against these leaks). BUT our folder-name label builder marked the whole tree fake (36.8% of "fakes" were real) — labels live in its metadata `target` field. Contains its own small ddpm folder (leaks the held-out generator into train if split naively) and inpainting/tampered generators (lama, mat, glide-in, palette). Ships COCO val2017 under Real/coco — the official real class. | Use metadata `target` for labels; route ddpm + tampered to TEST only; drop COCO-val2017 rows from train/val. |
| **LSUN church / bedroom (HF tglcourse/lsun_church_train etc.)** | Native short side 256 (measured via HF rows API). Reals at the diffusion fakes' scale. | Needed for content balance (see 3) — church was 27K real / 0 fake, bedroom 0 real / 21K fake before. |
| **CIFAKE** | 32px, single generator, its own resampling shortcut. | Toy only; numbers do not transfer. |

Final honest corpus (`docs/DATASET.md`): canon2 — 315K train / 40K val / 104K test, every image
a 176x176 PNG native crop, ddpm (21K) and tampered (27.6K) test-only, content mirrored per split.

## 2. Why "just resize everything" does not work (we proved it twice)
- Any DETERMINISTIC size mapping keeps a fingerprint: if classes differ in native size, either
  the output size or the resize FACTOR correlates with the label, and heavy resampling leaves
  a physical signature (detail density, smoothing). A resize-to-512 wrapper scored 0.50 on our
  test set and a size-poisoned model scored 0.27 (inverted) on honest data.
- Augmentation with random FACTORS does not help either: multiplication preserves the ratio
  between classes (256->128 vs 200->100 are still 1.28x apart).
- What works: **seeded random CROP at native resolution** (`scripts/canonicalize.py`) — a crop
  is subtractive, both classes come out identical size, zero resampling. For oversized eval
  images: downscale-only into the real class's size band, then crop (deflates at worst).
- Tiers: near-overlap sizes (<~1.5x) -> crop/random-band is adequate; large gaps (200 vs 1024)
  -> NO transform is trusted, fix the DATA (re-source the class at native resolution).
- Training data must be size-clean too: the model LEARNS the shortcut in training; eval only
  reveals it. Our patch model learned "duplicate crop tokens = real" because 200px reals took a
  different code path (upscale) than 256px fakes.

## 3. Other confounds we hit
- **Content skew:** ddpm fakes were churches+bedrooms, train had church=100% real and
  bedroom=100% fake. Mirror subject distribution across splits (LSUN reals added per split).
- **Compression history:** reals have >=1 JPEG generation, fakes are born PNG. Hunted it
  (jobs 41/42): our models turned out compression-invariant (0.985 -> 0.986/0.984 with equalized
  JPEG history), so NOT a leak for us — but check yours (`run_data_eq.sh` equalizes it).
- **Dumb-separable generators:** sfhq / face_synthetics / stargan (+cips) are separable by
  colour or file size alone. Exclude them from headline claims.
- **Held-out-generator leakage:** the held-out generator (ddpm) reappeared via a second
  dataset's folder. Enforce holdout by generator NAME across ALL sources.
- **Tampered/inpainted images in training** = label noise under cropping (a crop can land on
  the untouched region). Test-only.
- **Noise "paradox":** adding heavy noise gave AUROC 1.00 — pure artifact of the size path.
  Died at 0.335 on honest data.

## 4. What honestly generalizes (PE-Core-L14-336 fine-tuned, 316M params, canon2)
- Unseen generator within a SEEN family: ~0.96 (ddpm holdout 0.985 clean / 0.970 mean-TF).
- Unseen generator FAMILY (leave-one-school-out, all diffusion removed from train+val):
  GENERAL 0.716 — ddpm 0.845/0.811, glide 0.75, stable_diffusion 0.65, official DALL-E 0.66.
  Report both numbers, labelled. Cross-family generalization is the real unsolved problem.
- Seen schools (GAN/token) 0.96 clean. vqvae was our hardest honest cell (0.66-0.85 across
  models). Older honest baselines: frozen CLIP ViT-L + linear ~0.86 and transform-flat; ResNet-50
  random-size-crop mean-vote GENERAL 0.79; PE > ResNet by a wide margin on the same data.
- Multi-crop mean-vote beats whole-image scoring (27-grid 0.992 vs whole 0.985 on an unseen set);
  grid vs 100 random crops = statistical tie; trimmed mean +1.2 pts at 5% FA. Attention over
  patches ~= voting once the size artifact was removed.
- Train-time augmentation mirroring the eval grid: mean transformed AUROC +0.06 at -0.01 clean.
  Extra blur-heavy augmentation HURT on honest data (-0.03 official mean-TF) — rejected.
- Worst cells are always low-pass (blur_s2.0, resize_0.25x = blur): detectors read texture.
- Fixed threshold trap: at cut-off 0.15 the model flagged 10-27% of corrupted COCO reals; pick
  thresholds on VAL per deployment, report FPR under corruption.

## 5. Engineering traps (each cost us hours)
- Slurm: set `--cpus-per-task` AND `--mem` explicitly (defaults: 1 CPU, or first job hogs all
  RAM and blocks the second GPU). NEVER set CUDA_VISIBLE_DEVICES inside a Slurm job (cgroup
  renumbering -> silent CPU fallback; check the printed device line).
- torch.load needs `weights_only=False` for checkpoints containing numpy/objects (new default).
- Resume state: a stale `.state` file made a job "train" zero epochs and report the old val
  twice. Delete `.state` when starting a fresh run; watch for identical val across epochs.
- Class-sorted manifests + `--limit` head truncation = all-one-class subsample (NaN AUROC).
  Always seeded-shuffle before limiting.
- Silence != hang: a 5-model ensemble eval "froze" for an hour — it was just slow. Print per-batch
  heartbeats; `py-spy dump --pid` before killing anything.
- Keep checkpoints OFF the compute box (push to HF Hub / object storage after every run). Our
  server died with every checkpoint on it, one day before the deadline.
- Commit results JSON/tables to git as they land; docs updated with every change (the docs
  are the only reason this handoff exists).

## 6. Checklist before you quote a number
1. Manifest passed shortcut_audit + size_audit + canary/content audits (strict).
2. Held-out generator absent from train+val across ALL sources (grep by name).
3. Official / eval-only slices never in train/val.
4. Per-generator table inspected; any row >= 0.99 explained or excluded.
5. Clean vs mean-transformed vs worst-condition reported together; threshold chosen on val.
6. State whether the number is unseen-generator-seen-family or unseen-family.

## 6. Process lessons — every one of these was a miss caught by Thinh, not by a gate

Written 2026-08-31 during the canon6 rebuild. Each entry is a mistake that was actually
made in this repo, what it cost, and the thing that now prevents it. Kept because the
technical lessons above were already written down and the process ones still bit us.

### 6.1 The last run's config is not "the recipe"
canon6 was launched with `--stack-aug 0` because that is what `run_canon5.sbatch` passed.
It was the BASELINE arm; the arm that turned stacking on (`canon5_stack`, job 194) was
queued when the server died and never ran. Reading the shipped script told us what was
last executed, not what was intended.
**Prevention:** before copying a config, check it against the problem statement and the
queue/plan, not just the last script that ran. `--stack-aug 0.4 --stack-max 6` is the
canon6 setting and `run_canon6.sh` states why in a comment.

### 6.2 Keep the source text verbatim and separate from your reading of it
`docs/TRACK5_BRIEF.md` was a condensation with interpretation folded in — it asserted the
source "settles" that stacked transforms are in scope. The actual text says only "a subset
of the following augmentations" and is silent on how many per image. A defensible inference
had hardened into a quotation, and there was no clean copy to check it against.
**Prevention:** `docs/TRACK5_BRIEF_ORIGINAL.md` is the verbatim text, marked do-not-edit.
`TRACK5_BRIEF.md` is labelled as interpretation. Any doc that paraphrases a source needs
the source next to it.

### 6.3 "A subset" bounds WHICH, not HOW MANY
Reading it as "one transform, maybe two" under-covered the brief. A subset of six transform
families can be any size up to six, and the background text ("compressed, cropped, reposted")
describes chains.
**Prevention:** `EXTRA_GRID` now runs stack depths 2-6, and training draws depth over
2..`--stack-max`. The robustness table reports the whole depth curve so the reading is
visible rather than assumed.

### 6.4 Read the deliverables as a spec, not as background
The brief names literal required fields — development tools, libraries/frameworks, datasets —
and the word "compact" for the robustness summary. `docs/REPORT.md` had zero mentions of
development tools or of PyTorch/timm/scikit/Pillow.
**Prevention:** `scripts/robustness_table.py` collapses the 15 conditions into the SIX
families the brief itself tabulates, because compact is a requirement and not a style note.

### 6.5 Run the whole gate list, every time
CLAUDE.md binds every manifest to label_provenance + shortcut + size_audit. canon6 was
audited with four gates and `size_audit` was simply skipped — three had passed and it felt
covered.
**Prevention:** `python -m scripts.audit_all --prefix data/manifests/canon6` runs all of
them and prints one verdict table. Never run them one at a time from memory again.

### 6.6 An audit that passes may be blind, not clean  ← the expensive one
After canonicalization every image is a 176x176 PNG. `shortcut_audit` and `size_audit` both
read the CANONICAL files, so width/height/format are constant and the only feature left is
file size. They are structurally incapable of seeing native size.
`canon_unseen6` passed shortcut_audit at 0.617 ("mild") while:
  - three of its five native-size buckets contained NO FAKES AT ALL,
  - reals ran to 7712 px native against fakes capped at 1024 px.
"Big => real" was perfectly learnable, and the shrink-to-320 factor leaves a physical trace
(section 2). A pooled AUROC over that set would partly have been a measurement of image size,
and it was about to be reported.
**Prevention:** `audit_all` checks the NATIVE size distribution from the manifest's `long`
column and says in the output that the pixel audits are blind to it.
`scripts/size_matched.py` re-reads any evaluation per bucket from `scores.npz` and prints how
much the unmatched number was inflated. For an eval set with one-class buckets, the
size-matched number is the only one that may be quoted.
**General form: ask what each gate physically cannot see, and check that separately.**

### 6.7 Deduplicate a benchmark against training before quoting it
`bitmind/DiffFace-Real` is built from the same public face pools canon6 trains on: 229 of its
1,500 images (18%) were images we had trained on, plus ~8% of mobius / realvisxl /
bm-diffusion. The precedent was already in this repo — the original unseen-64 set was 31%
duplicate rows and re-reading it on unique images moved the headline.
**Prevention:** `scripts/dedup_unseen6.py`, matched on ORIGINAL files (canonicalize takes a
per-path seeded crop, so the same source image via two paths yields two different crops and
survives a post-canonicalization check).

### 6.8 A benchmark can only be quoted if it can be rebuilt
canon4's headline (0.9955 AUROC / 94% caught) was measured on `randtest_eq`, whose 64 sources
were documented BY CATEGORY only and whose builder (`extract_randtest.py`) was never
committed. When the server died the number became permanently unverifiable.
**Prevention:** every fetch is now a committed script with repo ids and shard slices
(`scripts/get_ext.py`, `scripts/build_unseen6.py`). If a number matters, its data must be
reproducible from the repo alone.

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

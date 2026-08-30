# Findings log — every observation, newest first

Rule (Thinh, 2026-08-30): every finding, observation, hypothesis and negative result goes here the
moment it lands, in plain language; an end-of-day summary block goes on top. ★ = interesting.
Numbers quoted here are on the audited sets described in docs/REPORT.md; "caught @1% FA" = share
of fakes flagged when the cut-off lets 1 in 100 real photos through as false alarms.

## 2026-08-30

### Day summary (written at end of day — pending)

### Findings
- **[hypothesis, Thinh] Why random crops can be worse than a grid even with 100–200 of them:** the
  randomiser can land on the same anomaly several times and on other regions never, so the average
  is weighted by luck, not by area. Fix: k shifted *partitions* of the image (deterministic, every
  pixel covered the same number of times) and/or weight the average per pixel or per area. Being
  tested (Slurm, `vote(t=m)` + `scripts/crop_dump.py` / `crop_agg.py`).
- ★ **Whole image (no cropping) is the worst way to read the image**, and worse the more it is
  shrunk: 82.0% caught @1% FA when shrunk to 168 px, 87.8% at 240 px, vs 90.8% for the 27-crop grid
  (same 64-source unseen set, canon4). The detector reads fine texture; shrinking erases it, and the
  model was trained on 112–168 px crops of 320-px images, so a whole image is a train/test mismatch.
- ★ **More crops does not help: the random-crop average is converged by 100.** 100 vs 200 random crops
  agree within 0.03 on 95% of images and flip the 0.15 verdict on 0.2% (37 / 17,064). Both: AUROC
  0.993, 88.0% / 88.6% caught @1% FA, 96.6–96.7% @5% FA.
- **Grid (27 crops) vs random (200 crops) is a statistical tie.** Same AUROC; grid +1.8 points @1% FA
  but paired bootstrap over the 900 reals gives [−1.3, +5.5] (the 1% cut-off rests on only 9 reals).
  The grid disagrees with the 200-crop average more than sampling noise would explain (95th-pct
  |Δ| 0.13 vs ≈0.06 expected for 27 random crops): the difference is systematic (grid always covers
  corners/edges), not noise; it favours the grid on Hunyuan 2.1 (54% vs 25%), Recraft v3 (74% vs
  59%), FLUX-2 Pro (10% vs 0%). Thinh's reading: the metric is saturated, so the difference is
  semantic (which regions get weight), not statistical. Kept the grid for now.
- **Inference rule matters far less than the data recipe.** Best-vs-worst rule here: 9 points
  @1% FA; canon3 → canon4 (data only, same rule) moved the same number 84 → 91.
- **Server: Slurm's GPU index is the reverse of nvidia-smi's** (Slurm IDX:0 = nvidia-smi GPU 1, PCI
  81:00.0). A `--gres=gpu:1` job lands on nvidia-smi GPU 1 first when both look free, and Slurm
  cannot see bare (non-Slurm) processes, so it schedules onto a card a teammate is already using.
  Job 77 collided this way and was cancelled within a minute. Rule now: every GPU job via sbatch;
  to pin a card, `--gres=gpu:2` + UUID assert at job start (run_r27.sbatch). docs/SERVER.md.
- **App cut-off 0.15 for canon4** = 1% false alarms on 900 never-trained reals; catches 90.8% of
  16,164 fakes from 64 never-trained generators (canon3 at its own 1%-FA cut-off: 84.0%).
- **canon4 holes (unseen generators, @1% FA):** FLUX-2 Pro 10%, Ideogram 51%, Hunyuan-Image 2.1
  54%, Seedream 3 60%. Strong on everything else (most sources 95–100%; Nano-Banana/Gemini 94%).
- **The ≥0.99 DALL·E numbers are not a palette/colour shortcut:** greyscale scoring 0.996 (canon4),
  channel-swap 0.998; the colour canary (0.78) is a property of the contest data, not of the model.
  No train/contest image overlap (perceptual hash 0/1).
- **Random unseen-generator test must be POOLED** (Thinh): per-generator AUROC only shows the model
  ranks within a generator; one cut-off across all sources at a fixed false-alarm budget is the
  number that means "general". Per-generator AUROC 0.999 can coexist with 10% caught (FLUX-2 Pro).
- **Mean-of-crops compresses scores toward 0** (canon4 more than canon3): a fake with a few
  suspicious regions averages low. Top-3 / max shift the scale but do not improve catch at equal
  false-alarm budget (44-set: mean 94.1% > 9-crop 92.8% ≈ top-3 92.6% > max 89.9%). The cut-off,
  not the rule, is what has to move.

## 2026-08-29 and earlier
See docs/PROGRESS.md (decision log) and docs/REPORT.md §2/§7 (findings + observation-list status):
size→label confound in both original benchmarks; shrink-first + per-bucket balance as the legality
rule; wild-set inversion (0/10 → 10/10) fixed by data, not by the model; crop-averaging +0.02 mean /
+0.03–0.04 on blur/resize; transformer ≫ CNN on identical data (0.964 vs 0.792); LOFO-diffusion
0.716 (never-seen *family* is the honest hard case); compression-history hunt negative; blur-boost
hurts on honest data.

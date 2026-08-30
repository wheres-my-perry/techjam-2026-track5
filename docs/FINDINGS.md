# Findings log — every observation, newest first

Rule (Thinh, 2026-08-30): every finding, observation, hypothesis and negative result goes here the
moment it lands, in plain language; an end-of-day summary block goes on top. ★ = interesting.
Numbers quoted here are on the audited sets described in docs/REPORT.md; "caught @1% FA" = share
of fakes flagged when the cut-off lets 1 in 100 real photos through as false alarms.

## 2026-08-30

### Day summary (written at end of day — pending)

### Findings
- **[brief, verified from source] "A subset of the following augmentations" limits WHICH transforms,
  not how many per image (Thinh) — stacking is in scope; single application is only the minimum. No
  hidden scored test set, no contest-defined metric or threshold.** The judged artefacts are the deliverables (Devpost text,
  README, YouTube demo, robustness summary table, error-analysis note) under five weighted criteria.
  Stacked conditions (5 fixed chains + seeded random 2-/3-stacks) now run beside the single grid.
- ★ **Unseen generators UNDER corruption hold up (job 144 probe: 300 unseen reals + 15 per generator
  × clean + 6 hardest corruptions, canon4 @0.15):** fakes caught 92.8% clean → 91.7–95.4% corrupted
  (corruption even helps a little — it removes the "too clean" look the model reads as real); reals
  flagged 1.0% clean → 3.2% mean, 6.3% worst (JPEG q30), noise 0.10 4.7%, resize ¼ 3.7%. Far milder
  than COCO (10% mean / 27% worst). Why: these reals are large (1024 px) — shrinking to 320 wipes most
  of the corruption before the model looks; COCO reals are 640 px and get shrunk less, WildFake reals
  (200 px) not at all. The vulnerable real population is small images, whatever the corruption.
- **Option C (log-odds mean) FAILS — no change (job 136 + offline):** at matched real-false-alarm
  rates on corrupted COCO reals (10% / 3% / 1%) it catches the same fakes as the plain mean on the
  DALL·E benchmark (99.1 vs 99.0%, 96.8 vs 96.6%, 92.5 vs 92.7%) and on the 64 unseen generators
  (90.1 vs 90.3%, 85.3 vs 85.4%, 81.3 vs 81.5%); worst-corruption real flags slightly worse (33 vs
  27% at the loose line). It rescales, it does not re-rank. Rest of the job cancelled to save GPU.
  Next: B (retrain).
- ★ **canon4_test at the fixed 0.15 (job 135, 8,000 held-out images, 32 known generators):** fakes caught
  74.5% clean / 73.7% mean over corruptions (dragged by tampering: SID-tampered 14%, inpainting 19%,
  LaMa 49%; pure generators 90–100%), real photos flagged **5.8% clean / 10.6% mean / 17.2% worst**
  (resize ¼). The real false alarms are concentrated by SOURCE: WildFake reals (200 px web images)
  35.6%, ArtiFact reals 14.4%, versus large ext reals 1.3%, LSUN 0–1%. So the model's weak real side
  is *small / low-detail* photos — the same failure as corrupted COCO reals (blur, ¼ resize, noise
  all remove detail). One cause, two symptoms.
- **Oversight (Thinh, 2026-08-30 ~17:00): the 64-source unseen test was never run under the
  corruption grid**; only clean native-size images. Fixed as a small probe (300 reals + 15 per
  generator, clean + 6 hardest corruptions, ~13 min/model; `--conditions` in src.evaluate), job 144
  for canon4 mean + logit, and inside run_B.sh for every candidate.
- **One ranking over everything (Thinh's "AUROC over the whole dataset"), canon4, cut-off 0.15
  (`scripts/pool_auroc.py`):** clean-only pool of 18,274 images — judges' benchmark 1,200 (6.6% of
  the pool), 64-source unseen 17,064 (93.4%), wild 10 (0.1%) — **pooled AUROC 0.990**, 90.8% caught,
  1.6% reals flagged. Adding the 14 corrupted copies of the benchmark (35,074 rows: benchmark 51.3%,
  unseen 48.7%) — **pooled AUROC 0.979**, 94.0% caught, **8.6% reals flagged**. The pooled AUROC is
  below every group's own AUROC (0.9999 / 0.992 / 1.0): groups do not share a scale — corrupted
  COCO reals (0.15–0.3) rank above the hard unseen fakes (FLUX-2 Pro ~0.08) — which is exactly the
  fixed-cut-off problem in ranking form. canon4_test joins the pool when job 135 finishes.
- ★★ **At the fixed cut-off 0.15 the judges' benchmark has a REAL-side problem under corruption
  (job 135, canon4, DALL·E-3 vs COCO val, 1,200):** fakes caught 100% clean / 99.0% mean over
  corruptions / 97.0% worst, but real photos wrongly flagged 2.9% clean / **10.2% mean / 26.7%
  worst** (resize ¼ 26.7%, noise σ0.10 24.7%, blur σ2 20.8%, noise σ0.05 12.7%, resize ½ 11.1%).
  Heavily blurred/downscaled/noisy reals drift up toward the line; DALL·E fakes sit near 1 and
  never come down. AUROC hid this completely (0.9855 worst). Cut-off sweep (all sets at once):
  0.15 → unseen-gen 90.4% / COCO reals mean-TF 10.2% flagged / wild 5/5; 0.30 → 85.9% / 3.3%
  (worst 8.8%) / 3/5, DALL·E mean-TF 96.7%; 0.50 → 81.0% / 0.9% / 2/5. One line cannot serve both
  "unseen family scores 0.2–0.35" and "corrupted real scores 0.15–0.3": those two populations
  overlap in score. Fix must come from the model's real side (train on more corrupted reals) or
  from choosing the line for the judged benchmark (~0.3) and accepting the unseen-family loss.
- ★ **"Boost blatant AI-ness" aggregations (Thinh's f(a_i) idea) make it WORSE, measured on the
  saved per-crop scores (17,064 images):** power mean p=2/3/5: 89.9/89.5/88.2% @1% FA (mean:
  90.8%); softmax-weighted β=5/10: 86.1/81.5%; max 85.0%; tempered noisy-OR 89.9%; quantile-90
  88.9%. They are also LESS stable under crop resampling (verdict flips: softmax 1.80%, max 1.45%
  vs mean 1.18%). Why: reals also have occasional loud crops (5% of Open Images reals have a crop
  > 0.9) so any rule that listens to the loudest crop buys false alarms; and the hard fakes have no
  loud crops to boost (FLUX-2 Pro: 4% of crops > 0.5, only 10% of images have any crop > 0.9;
  Hunyuan/Ideogram/Seedream ~45–54%). The one f that helps is the log-odds mean (average logit,
  then sigmoid): AUROC 0.9936 (best), 97.4% @5% FA (best), 90.3% @1% FA (tie), and the most stable
  rule (0.90% flips, reals 0.13%). But it does not rescue near-line unseen fakes — it pushes them
  down (Gemini 0.19 → 0.04) because most of their crops look real; it stretches the scale around
  the reals. Closeness to the line for unseen families is a model-knowledge limit, not an
  aggregation problem.
- **[process, Thinh] A product has ONE cut-off; every number must be read at it.** Until now the
  robustness tables (DALL·E benchmark, canon4_test) reported AUROC plus accuracy at a threshold the
  harness picked by Youden's J on *that set's own clean scores* (0.268 on DALL·E, 0.113 on
  canon4_test) — two different lines, each tuned on the set it is reported on, i.e. a mild leak and
  not a product number. Only the 64-source unseen test, the wild set and DIV2K were read at the
  shipped 0.15. Fix: `src.evaluate --threshold 0.15` (fixed; reports fakes caught / reals flagged /
  accuracy per condition and per generator; saves scores.npz so any cut-off is re-readable). Job
  135 re-reads the DALL·E benchmark and canon4_test at 0.15.
- **Wild set (5 iPhone photos + 5 Gemini images) at the shipped cut-off 0.15: 10/10.** Reals 0.000–0.041,
  Gemini 0.191–0.776; nearest miss on each side: real 0.041 vs fake 0.191 (margin 0.15 around the
  cut-off). At 0.5 it was 7/10 — the same scores, a different line.
- ★ **Even-coverage tilings and per-pixel weighting (Thinh's proposal) do not beat the grid — the
  crop LAYOUT is saturated.** Same 64-source set, canon4, identical crops re-aggregated offline
  (job 80): every layout × every rule lands at 88–91% caught @1% FA and AUROC 0.991–0.993.
  Grid 27 / mean 90.8%; t=1 partition (20 crops) 89.6%; t=2 (44) 89.4%; t=3 (75) 90.0%; per-pixel
  weighting of the grid 89.5% (−0.3, CI [−1.4, +1.4]); area-weighted 90.0% (±0). Coverage evenness
  cannot be achieved by layout on real aspect ratios (a 320×213 image fits one row of 168-tiles;
  clamped tiles overlap: min/max coverage 3/9 for t=1 vs 3/15 for the grid), and making the
  weighting exactly even per pixel changes nothing. Conclusion: what matters is *that* the image is
  read in native-scale crops, not *where* they are placed.
- ★ **A trimmed mean (drop the 10% highest and 10% lowest crop scores, ~3 of 27) is the only rule
  that is reliably better than the plain mean** — not at the 1%-FA operating point (+0.1, CI
  [−0.3, +0.7]) but at 5% FA: +1.2 pts (95.9 → 97.1%), CI [+0.7, +1.7], and best AUROC of all
  (0.9932). Median gives the same +1.2 at 5%. The gain comes from hard generators where a few
  extreme crops drag the mean: Hunyuan-Image 2.1 77 → 100% @5% FA, FLUX-2 Pro 89 → 99%, GPT-4o
  88 → 94%, Recraft v3 93 → 96%. Top-3 is worst everywhere (85–88%). Not shipped (no gain at the
  operating point; would need a new cut-off + official re-eval); recorded as a free option.
- ★ **27 random crops DO carry real per-image noise (Thinh was right on this point):** the same
  images scored with two different random seeds (27 crops each) differ by ≥0.10 for 5% of images
  (99th pct 0.16) and 1.0% of verdicts at cut-off 0.15 flip on luck alone. The pooled metric does
  not move (both seeds: AUROC 0.992, 87.5% caught @1% FA) — the aggregate is saturated, the
  per-image verdict is not. At 100 crops the noise is gone (100 vs 200: 95th pct 0.03, 0.2% flips).
  The shipped 27-crop GRID has zero seed noise (deterministic) but sits further from the converged
  200-crop average (95th pct 0.13, 1.6% flips) than the noise itself — its layout is a systematic
  bias relative to uniform sampling, one that currently helps slightly (+1.8 pts @1% FA, n.s.).
  Rule of thumb: ~1 in 100 verdicts near the cut-off is decided by crop luck at 27 random crops.
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

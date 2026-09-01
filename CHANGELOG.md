# Changelog

Dated development history. Entries describe the repository at that point in time and include both
shipped changes and experiments. Current behavior is documented in [README.md](README.md) and
[ARCHITECTURE.md](ARCHITECTURE.md); retained superseded reports live under
[archive/docs/](archive/docs/).

## 2026-08-31 (server loss — corpus and checkpoint rebuilt from scratch on a new box)
- **mio03 died with every checkpoint and all of canon5 on it.** GitHub had only code and text, so
  there was nothing to restore: no weights existed anywhere off-box. Rebuilt on a fresh vast.ai
  instance (RTX 5090, 500 GB) rather than shipping a pre-confound Aug-26 checkpoint.
- **scripts/get_wildfake.py — label matching rewritten to full-path suffix resolution.** The 08-30 fix
  guarded only the csv's TOP-LEVEL folder, which stops fakes matching real photos but not collisions
  inside Real/: church/imagenet/ffhq/afhq/celebahq all name their files img000000.jpg. Measured on the
  live tree, that basename alone resolves to SIX different files. lookup() now takes the longest path
  suffix that exists on disk and reports AMBIGUOUS rather than guessing.
  Verification: all 8 downloaded sources resolve 613,195 rows with ZERO ambiguity, while the four
  not-downloaded GAN/VQVAE csvs (styleGAN 80,000 · VQVAE 55,000 · BigGAN 10,000 · starGAN 9,995 =
  154,995 rows) are all caught as ambiguous. Those are exactly the rows that became "fakes pointing at
  real photos" in canon2..4 — the bug reproduced live and blocked.
- **scripts/build_canon6.py** — assembles the corpus with the protocol enforced in code, not assumed:
  generator hold-out keyed on NAME across all datasets (ddpm ships in both WildFake and ArtiFact),
  partial edits test-only, splits over source files, per-native-size-bucket class balance.
- **scripts/get_ext.py / extract_artifact_subset.py** — record the HF repo ids and shard slices for the
  large-image sources, which had only ever been fetched by hand on the dead server. ArtiFact is
  extracted in two phases (metadata, then only sampled images) instead of unpacking 2.5M members.
- **canonicalize.py** now emits the native `long` side (measured before any resize) so the bucket gate
  needs no second pass.
- **Benchmark integrity: 184 COCO val2017 rows found inside ArtiFact's coco folder and dropped** before
  training. That is the judges' real class; ArtiFact ships it under Real/coco (same class of leak as the
  487 rows that reached canon2_train).
- **canon6**: train 124,792 (62,396/62,396, 26 fake generators) · val 15,598 · test 131,684 (33 generators,
  incl. held-out ddpm 30,896 + DeepFloyd-IF + partial-edit sets). Gates: label provenance CLEAN, bucket
  CLEAN, metadata-only 0.6292 (mild, canon5 was 0.6313), style canary 0.6875 (FAIL, canon5 was 0.6795 —
  same known property; checked on the checkpoint via greyscale/channel-swap instead).
  Adding 19,640 COCO train2017 reals moved metadata 0.6585 FAIL -> 0.6292 and filled the 513-768 bucket
  (37 -> 4,227 pairs), which is what let the 640px ELSA fakes train instead of being dumped to test.
- **Training runs the canon5_stack recipe (job 194, queued on mio03 but never executed): --stack-aug 0.4**,
  i.e. 40% of samples get a random 2-or-3 transform stack from the brief's grid, both classes. canon5's
  own `--stack-aug 0` was the baseline, not the intended setting.
- **docs/TRACK5_BRIEF_ORIGINAL.md** — the verbatim problem statement is now in the repo as the single
  source of truth; [archive/docs/TRACK5_BRIEF.md](archive/docs/TRACK5_BRIEF.md) preserves our former
  interpretation. The condensation had claimed
  the source "settles" that stacking is in scope; the original text does not say that.
- NOT reproducible: the unseen-64 benchmark (randtest_eq). The docs describe its sources by category
  only and extract_randtest.py is not in the repo, so canon4's "0.9955 AUROC / 94% caught" cannot be
  re-measured. Unseen-generator claims now come from canon6's own held-out generators.

## 2026-08-30 (morning — crop-count / aggregation study)
- vote(t=m) even-coverage tilings; scripts/crop_dump.py (per-crop scores + boxes, one GPU pass) and
  scripts/crop_agg.py (offline rules: mean/size/area/pixel/median/trim10/top3). Job 80: all layouts x
  rules within +-1.5 pts of the shipped grid @1% FA; trimmed mean +1.2 @5% FA (significant). Kept grid+mean.
- 27 random crops, two seeds (job 79): pooled identical, 1.0% of per-image verdicts flip on seed alone.
- src/predict.py then defaulted to the shipped policy + --threshold 0.15 + label field.
  [archive/docs/FINDINGS.md](archive/docs/FINDINGS.md) was started. (The current threshold is 0.5.)
- src/model.py CropVoteModel: `r=N` (N seeded random crops instead of the grid) and `s=seed` spec keys.
- Same 64-source unseen set, canon4: 27-grid 0.992 / 90.8% @1%FA; 100 random 0.993 / 88.0%; 200 random
  0.993 / 88.6%; 1 centre crop 0.988 / 88.6%; whole image 168px 0.985 / 82.0%, 240px 0.987 / 87.8%.
  100 vs 200 crops agree on 99.8% of verdicts -> more crops is converged; grid-vs-random is a tie
  (paired bootstrap +1.8 [-1.3,+5.5]). Kept the 27 grid. REPORT §4.

## 2026-08-30 (night — canon4 result, random unseen-generator test)
- Job 76 canon4: GENERAL 0.977, DALL·E 0.996 mean-TF, DeepFloyd (test-only) 0.942, MJ-v6 held-out 0.999.
- scripts/random_gen_test.py: 39 (then ~50) never-trained generator sources x 300 images vs 900 never-trained
  reals at native size; per-generator + POOLED AUROC and catch at fixed false-alarm rates; --save scores.
- app.py + score_dir.py -> canon4.pt, threshold 0.15 (pooled unseen-generator test: 94% caught at 1% FA vs 83% for canon3).
- Disk hit 100%: removed COCO train zip and already-extracted parquet shards (re-downloadable).

## 2026-08-30 (early — canon3 final numbers, app on canon3)
- Job 67 canon3 (328K rows, 4 ep): GENERAL 0.970; DALL·E 0.996/0.985/0.964 raw, 0.997/0.988/0.969
  dedup, 0.995/0.986/0.967 style-matched; WILD 10/10 at 0.5. REPORT §5.1b.
- app.py + scripts/score_dir.py default -> vote(L=320)+pe_ft:outputs/pe_ft/canon3.pt, threshold 0.5.
- canon4 manifests (iteration B): merge_ext.py --raw/--ext/--test-only; run_canon3.sh evaluates ${P}_test;
  all ext sources extracted (raw_ext_all.csv) + COCO train2017 12K originals for the 640 bucket. Job 76.

## 2026-08-29 (evening — small-trial verdict, pe_seg, report)
- Job 64 (canon3s small trial): wild set flips from inverted to ranked (0.96/1.00); DALL-E 0.974 mean.
- Job 65 pe_seg: per-patch localisation on SID masks, held-out patch AUROC 0.984; heat-maps in docs/figures.
- [archive/docs/REPORT.md](archive/docs/REPORT.md): then-current submission write-up (findings,
  recipe, results, observation-list status).
- pe_ft --real-weight / --limit-train; generate_hard_fakes keeps native size by default.

## 2026-08-29 (afternoon — wild test, large-image expansion)
- app.py: gradio prototype (drop image -> P(AI) + per-crop map + transform picker); HEIC support
  (iPhone .jpeg files are HEIF) in src/data.load_image via pillow-heif.
- scripts/wild_eval.py: frozen data/hack held-out set (phone photos + Gemini). canon2 model: 0/10.
- Large-image expansion: scripts/build_ext_manifest.py (SID_Set, CelebA-HQ, AFHQv2, Open Images,
  FFHQ, Midjourney v6, ELSA_D3; original bytes kept), canonicalize.py --long (shrink by long side,
  worker pool), scripts/merge_ext.py (canon3; per-bucket class balance, excess -> test, tampered ->
  test), scripts/bucket_audit.py (gate). CropVoteModel vote(L=320) mirrors --long at inference.
- run_canon3.sh/.sbatch: gates (strict) -> pe_ft train -> canon3_test/official/GENERAL/WILD.

## 2026-08-29 (afternoon — leave-one-school-out verdict)
- Job 43 (run_lofo.sbatch diffusion pe_ft): PE with every diffusion generator removed from
  train+val scores GENERAL 0.716 (ddpm 0.811 mean-TF, official 0.620) vs 0.964 when sibling
  diffusion models are in training. Within-family transfer is strong (0.96), cross-family is
  weak (0.72). The 0.964 must be reported as "unseen generator, seen family".

## 2026-08-29 (night — data audit, content matching, random-size crops, pe_ft)
- ArtiFact labels were WRONG: builder labelled by folder name but the tree is ArtiFact/{Real,Fake}/
  <source>/, so all 2.5M files went in as fake — 36.8% of the sampled "fakes" were real photos
  (32K LSUN, 10K COCO...). Labels now come from metadata.csv `target`; reproduces the published
  964,989/1,531,749 exactly. Corpus 47% real (was 19%). All job-28/30 numbers void.
- ddpm held-out contamination: ArtiFact's own ddpm folder was split 80/10/10 → 710 in train.
  merge_manifests routes it (and tampered: lama/mat/generative_inpainting/palette/glide-in) to test.
- Content matching (new standing rule): ddpm fakes are LSUN church+bedroom; train had 27K always-
  real churches and 21K always-fake bedrooms. LSUN church 45K→15K (test-weighted) + LSUN bedroom
  25K added (mirrors fake bedrooms per split). scripts/content_audit.py flags one-sided subjects.
- New gate scripts/canary_audit.py (deliberately dumb pixel models must score ~0.5); both audits
  gained --strict so an afterok chain cannot train on a FAILed manifest (before: print-only).
  Official benchmark FAILS the canary (colour 0.755/hist 0.764) — recorded as a standing caveat.
- run_data.sh guards the 31.7GB re-download; run_canon2.sh deletes the stale .state (job 30
  silently resumed from epoch 7 and trained ZERO epochs — the tell was val 0.9307 twice).
- Random-size crops (Thinh): src/crops.py shared by training (size per batch in collate) and the
  vote+ wrapper (ladder over the same range); never upscale (old wrapper upscaled 176→224).
- New approach src/approaches/pe_ft: facebook/PE-Core-L14-336 (316M) full fine-tune at 112-168px
  via pos-embed interpolation, sides snap to 14. Predictions were recorded in an unretained
  GENERATOR_MATRIX; [archive/docs/DATASET.md](archive/docs/DATASET.md)
  written for teammates.

## 2026-08-28 (late night — dev environment moves to the server)
- Git topology reversed (Thinh): server clone is now primary and pushes; Mac clone becomes the
  read-only mirror. CLAUDE.md rule rewritten; its former DECISIONS note is not retained.
- CLAUDE.md itself committed (was untracked) so agents starting on the server get the briefing.

## 2026-08-28 (late night — data expansion approved)
- Thinh approved the balanced corpus: ArtiFact subset (150K real + 150K fake @200, 25 generators
  incl. TOKEN family; size+JPEG randomized by its authors) + LSUN Church 45K reals @256 (pairs
  with ddim/ddpm) + existing WildFake. ~430K images total, ~34GB new.
- New: scripts/{get_lsun,build_artifact_manifest,merge_manifests}.py; run_data.{sh,sbatch} (cpu:
  download->canonicalize->merge->audit gates) chained via Slurm dependency to run_canon2.{sh,sbatch}
  (gpu: retrain resnet on canon2 -> evals). The former DATA_CANDIDATES note is not retained.

## 2026-08-28 (night — canonical protocol)
- scripts/canonicalize.py: seeded-random per-image resize into overlapping band, one filter/one
  format both classes; verified to produce CLEAN shortcut audits. run_canon.{sh,sbatch}:
  canonicalize wildfake train/val/test (band 176-200) + official_v2 (band 320-512), audit gates,
  retrain resnet_ft at crop 160 -> outputs/resnet_ft/canon.pt, evals on canon manifests.
- Decision (Thinh): training stays purely-generated vs purely-real; tampered data excluded
  (crops would cut off edited regions -> label noise), reserved as future stress-test.

## 2026-08-28 (evening)
- FIX DONE results: honest official_v2 numbers std+patch_relation 0.886/0.871, std+vote+resnet
  0.889/0.878; raw patch_relation inverted (0.272); noise+ trick dead (0.335). wf_test itself
  size-structured (reals+GAN+vqvae 200px matched, diffusion 256px leaky). Full size-canonical
  data protocol scheduled: single preprocess pipeline for all data, remanifest, retrain.

## 2026-08-28 (afternoon — confound response)
- Found official_val size confound (200x200 reals vs 1024+ fakes). Added std+ wrapper
  (src/model.py), scripts/size_audit.py, scripts/rebuild_official.py (-> official_v2.csv with
  original-resolution COCO val2017 reals), run_fix.{sh,sbatch}. All prior official-benchmark
  numbers are marked suspect in docs until re-measured on official_v2.

## 2026-08-28 (day shift)
- Stacked ensemble built (src/approaches/stacked/, model name `stacked`): tiny classifier over
  member scores, fit on augmented val only. noise+ wrapper added to src/model.py (obs #12
  kill-test). real_manifold features NaN-proofed (flat crops). run_stack.{sh,sbatch} runs
  reruns + noise tests + stacker train/eval; weights auto-detected by newest-file glob.

## 2026-08-28 (morning)
- Night shift 2 verdict: patch_relation champion (official 0.958/0.935/0.817), blur-boost resnet
  second (0.952/0.932/0.826), spectral killed. Retained summary:
  [archive/docs/PROGRESS.md](archive/docs/PROGRESS.md); per-approach and GENERATOR_MATRIX files are
  not retained. Known reruns: vote+clip_linear, vote+cnn (weight filename fix),
  vote+real_manifold (NaN on flat crops).

## 2026-08-28 (night shift 2b — attention)
- New approach: src/approaches/patch_relation/ (01 stage 2) — frozen resnet_ft trunk + 2-layer
  transformer relation head over 3x3 native-res patch grid; registered as `patch_relation`.
  Sharded resumable embedding cache; self-contained checkpoint. Job: run_attn.{sh,sbatch} (GPU 2).

## 2026-08-28 (night shift 2)
- New approach: src/approaches/spectral/ (03) — FFT artifact features + logistic head,
  CPU-only, registered as model name `spectral`. Smoke-tested end-to-end on synthetic
  checkerboard/upsample fakes (harness integration verified).
- resnet_ft train gains --blur-boost (extra blur/downscale augmentation targeting the
  measured blur/resize weakness); new checkpoint target outputs/resnet_ft/wf_blur.pt.
- Night jobs: run_night2.{sh,sbatch} (GPU: blur retrain + vote+clip_linear/vote+cnn evals),
  run_spec.{sh,sbatch} (CPU: spectral train+eval, vote+real_manifold).

## 2026-08-28
- Night eval batch (Slurm job 10) complete: 7 evals at --limit 1200 across wf_test + official.
  Verdict retained in [archive/docs/PROGRESS.md](archive/docs/PROGRESS.md); the former per-family
  GENERATOR_MATRIX file is not retained.
- Crop-voting wrapper (vote+<model>) validated: official benchmark resnet_ft 0.207 (inverted) ->
  vote+resnet_ft 0.938 clean / 0.913 mean transformed. wf_test clean 0.954.
- Workflow: back to Slurm (sbatch/squeue/scancel) per Thinh; the former SERVER/CHEATSHEET notes and
  project-conventions skill were reverted. Server had no rsync, so results were pulled via
  tar-over-ssh.

## 2026-08-27

- Kill-resilient training: per-epoch resume state for cnn/resnet_ft, sharded resumable CLIP
  embedding cache (atomic shard writes), retry-loop sbatch wrappers. Smoke-tested (shard resume,
  cache keying, retry flow).
- Third approach `resnet_ft`: ImageNet-pretrained ResNet-50, fully fine-tuned, registered.
- WildFake pipeline completed on GPU server: 250K images indexed, manifests built
  (5 train generator families, ddpm held out, official benchmark 8,843+4,998 verified exact).
- `get_wildfake.py` rewritten CSV-driven (label_csv_files authoritative); selective zip extraction
  (`--extract-filter`), `--delete-zips`, `--official-val`; ModelScope CLI/API compat fixes.
- ARCHITECTURE.md + this changelog added (team-lead request: agent-maintained docs).
- First real-data result: scratch cnn (w64, aug, crop224, 15 ep) val AUROC **0.811** on 5-generator
  WildFake — vs 0.98 on single-generator CIFAKE. Capacity/diversity gap confirmed.

## 2026-08-26

- Repo bootstrapped (org `wheres-my-perry`, public), docs set (TRACK5_BRIEF/PROGRESS/DECISIONS/IDEAS),
  5 Claude agent skills in `.claude/skills/`.
- Evaluation harness: 15-condition contest transform grid, AUROC / balanced-acc@frozen-threshold /
  FPR@95%TPR, per-generator breakdown, error dumps, contest `predict` CLI, tests + synthetic smoke test.
- Approach structure `src/approaches/<name>/` (decision: folders over branches); `cnn` and
  `clip_linear` implemented.
- CIFAKE experiments: aug-vs-noaug robustness result (mean transformed AUROC 0.88→0.94, worst
  0.70→0.885 at clean cost −0.012) — core augmentation hypothesis validated. cnn w64: 0.961 mean
  transformed. clip_linear B/32 sanity run.
- Colab pipeline notebook (superseded by GPU server, kept as fallback).

## 2026-08-31 (late) — head shape, val augmentation, partial-edit experiment

- **MLP head 1024->64->1 shipped** over the single linear layer: +2.1 points of recall on the
  judges' set at an identical 1% false-alarm rate, +0.03 AUROC on the hack set, everything else
  held identical. `src/approaches/pe_ft/model.py:make_head`; checkpoints select their own head
  shape from the state_dict keys, so old and new weights both load with no flag.
- **`--val-augment` added to `pe_ft/train.py`.** Validation was scored on CLEAN images while train
  and test were augmented, so "best val AUROC -> save" was blind to robustness. Off by default for
  comparability with earlier checkpoints; on for any run that selects a checkpoint.
- **`src/evaluate.py`: `--conditions clean` no longer crashes.** With no transformed condition the
  `np.nanmin(tprs)` summary raised on an empty array AFTER every image had been scored and BEFORE
  scores.npz was written — a whole evaluation lost to a summary statistic. Transformed statistics
  now report None.
- **`--train-partial-edits FRAC` added to `scripts/build_canon6.py`** (experiment only) to measure
  what happens if partially edited images are TRAINED on instead of held out for test. Verified:
  with the flag off the builder reproduces `canon6_train.csv` byte-for-byte.
- **Negative result: augmentation-consistency loss.** Best clean val (0.9973), worst hack set
  (0.870). That unrestrained variant was not shipped; the restrained low-LR consistency variant was.
  Historical discussion: [archive/docs/REPORT.md](archive/docs/REPORT.md) section 4.3.

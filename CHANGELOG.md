# Changelog

Dated, shipped changes only. (Current status & next steps: [docs/PROGRESS.md](docs/PROGRESS.md);
design candidates: [docs/IDEAS.md](docs/IDEAS.md); decisions: [docs/DECISIONS.md](docs/DECISIONS.md).)

## 2026-08-30 (early — canon3 final numbers, app on canon3)
- Job 67 canon3 (328K rows, 4 ep): GENERAL 0.970; DALL·E 0.996/0.985/0.964 raw, 0.997/0.988/0.969
  dedup, 0.995/0.986/0.967 style-matched; WILD 10/10 at 0.5. REPORT §5.1b.
- app.py + scripts/score_dir.py default -> vote(L=320)+pe_ft:outputs/pe_ft/canon3.pt, threshold 0.5.
- canon4 manifests (iteration B): merge_ext.py --raw/--ext/--test-only; run_canon3.sh evaluates ${P}_test;
  all ext sources extracted (raw_ext_all.csv) + COCO train2017 12K originals for the 640 bucket. Job 76.

## 2026-08-29 (evening — small-trial verdict, pe_seg, report)
- Job 64 (canon3s small trial): wild set flips from inverted to ranked (0.96/1.00); DALL-E 0.974 mean.
- Job 65 pe_seg: per-patch localisation on SID masks, held-out patch AUROC 0.984; heat-maps in docs/figures.
- docs/REPORT.md: living submission write-up (findings, recipe, results, observation-list status).
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
  via pos-embed interpolation, sides snap to 14. Predictions in GENERATOR_MATRIX. docs/DATASET.md
  written for teammates.

## 2026-08-28 (late night — dev environment moves to the server)
- Git topology reversed (Thinh): server clone is now primary and pushes; Mac clone becomes the
  read-only mirror. CLAUDE.md rule rewritten, rationale in docs/DECISIONS.md.
- CLAUDE.md itself committed (was untracked) so agents starting on the server get the briefing.

## 2026-08-28 (late night — data expansion approved)
- Thinh approved the balanced corpus: ArtiFact subset (150K real + 150K fake @200, 25 generators
  incl. TOKEN family; size+JPEG randomized by its authors) + LSUN Church 45K reals @256 (pairs
  with ddim/ddpm) + existing WildFake. ~430K images total, ~34GB new.
- New: scripts/{get_lsun,build_artifact_manifest,merge_manifests}.py; run_data.{sh,sbatch} (cpu:
  download->canonicalize->merge->audit gates) chained via Slurm dependency to run_canon2.{sh,sbatch}
  (gpu: retrain resnet on canon2 -> evals). Verification evidence in docs/DATA_CANDIDATES.md.

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
  second (0.952/0.932/0.826), spectral killed. Details: docs/PROGRESS.md, docs/approaches/01+03,
  GENERATOR_MATRIX actuals. Known reruns: vote+clip_linear, vote+cnn (weight filename fix),
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
  Verdict in docs/PROGRESS.md; per-family actuals in docs/GENERATOR_MATRIX.md.
- Crop-voting wrapper (vote+<model>) validated: official benchmark resnet_ft 0.207 (inverted) ->
  vote+resnet_ft 0.938 clean / 0.913 mean transformed. wf_test clean 0.954.
- Workflow: back to Slurm (sbatch/squeue/scancel) per Thinh; SERVER.md, CHEATSHEET.md, and
  project-conventions skill reverted. Server has no rsync — results pulled via tar-over-ssh
  (command in CHEATSHEET.md).

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

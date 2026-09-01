# Error analysis

This note covers the shipped `canon6_AlowLR` checkpoint. Its clearest strength is detecting fully
generated images after common transformations. Its clearest measured weakness is detecting a small
generated edit inside an otherwise authentic photograph.

## Read the thresholds first

There is no one cutoff shared by every experiment in this document:

| evaluation | threshold convention |
|---|---|
| CLI and Gradio verdict | fixed product threshold of **0.5** |
| clean and transformed 900-image slices | separately calibrated within each slice to about 1% false alarms |
| stacked-transform ladder | one cutoff per model, calibrated on pooled clean + depths 1–6 reals and fixed across all depths |
| held-out test and localized-edit set | one set-specific cutoff calibrated on all reals in that set |

These calibrated cutoffs measure separation at a comparable operating point; they are not deployed
thresholds. AUROC is threshold-free.

## Main failure modes

| case | measured result | reading |
|---|---|---|
| Fully generated DALL·E images, six transform families stacked | 245 / 248 caught; 4 / 152 reals flagged | strong result on the contest reference subset |
| Localized edits, leak-checked set | 320 / 1,182 caught; 12 / 1,182 reals flagged | dominant measured weakness |
| Held-out mixed test, pooled across conditions | 73.7% recall; 1.00% false alarms; 0.9580 AUROC | includes difficult partial edits and a broader source mix |

The datasets and threshold policies differ, so these rows should not be compared as a single
leaderboard.

## False negatives: localized edits

On the dedicated edit set, the shipped model catches **27.1%** of 1,182 tampered images at 12 false
alarms among 1,182 real photographs. These examples are mostly authentic pixels with only one
region replaced, unlike the fully generated images that dominate training.

The inference policy may contribute. It averages up to 27 crop scores, so evidence confined to a
small region can be diluted by crops containing mostly authentic content. This is a plausible
mechanism, not a proven cause: we did not complete an aggregation or localization ablation for the
shipped checkpoint.

A controlled data experiment on the non-shipped MLP baseline supports a data-side improvement.
Adding partial edits to its training set raises edit recall from **23.3% to 72.1%**, while reference-
set recall changes from 97.0% to 96.7%, at each checkpoint's calibrated operating point. That
comparison changes training data on the same baseline; it should not be presented as an absolute
result for `canon6_AlowLR`.

Useful next experiments are a max or top-k crop aggregator, an explicit localization head, and
training with leak-checked partial edits while keeping a fully held-out edit source for evaluation.

## False positives and calibration drift

On the 900-image reference slice, separate calibration gives:

| slice | cutoff | false alarms | AI recall | AUROC |
|---|---:|---:|---:|---:|
| clean | 0.0650 | 4 / 326 | 574 / 574 | 0.9999 |
| 14 transformed conditions pooled | 0.2788 | 46 / 4,564 | 7,922 / 8,036 | 0.9995 |

The higher transformed cutoff shows that transformations move calibration even when ranking stays
strong. Because the two rows are recalibrated independently, they cannot be used to claim robustness
at one fixed threshold.

The stacked-depth experiment does hold one pooled cutoff fixed. For the shipped model, false alarms
rise from 0 / 152 on clean images to 4 / 152 after all six transform families are applied. That is a
small sample—one photograph changes the rate by about 0.7 points—but it makes the operational
trade-off visible.

The false-positive and false-negative contact sheets are examples for inspection, not estimates of
how often a visual pattern occurs:

- [Highest-scoring real photographs](../error_analysis/FP_real_called_AI.jpg)
- [Lowest-scoring generated images](../error_analysis/FN_dalle_called_real.jpg)
- [Ranked clean-reference cases](../error_analysis/worst.csv)
- [Held-out-test contact sheets and cases](../error_analysis/ship/)

## Trade-offs and limits

- **Robust global detection versus localized edits.** Crop averaging is useful for image-level
  stability, but it may suppress small edited regions. The causal contribution has not yet been
  isolated.
- **One deployment cutoff versus recalibration.** Set-specific calibration is useful for analysis,
  but a real system needs a cutoff chosen on a representative deployment mixture and monitored as
  sources and transformations change.
- **Optimization evidence.** The shipped low-LR consistency configuration is the strongest tested
  combination, but it changes both trunk learning rate and consistency weight relative to model A.
  The current comparison does not attribute the gain to either change alone.
- **Evaluation coverage.** The strongest stacked result uses 400 images from only DALL·E-3 and COCO.
  It should be repeated on independent real-image sources and fully held-out generator families.

## Provenance

- clean and transformed reference slices: [`logs/night2.log`](../logs/night2.log)
- held-out mixed test: [`logs/ship_eval.log`](../logs/ship_eval.log)
- localized edits: [`logs/ship_final.log`](../logs/ship_final.log)
- controlled partial-edit data experiment: [`logs/edits2b.log`](../logs/edits2b.log)
- stacked-depth run: [`logs/bench.log.part1-0113`](../logs/bench.log.part1-0113)

# Approach 05 — Test-Time Prediction Consistency

Status: unbuilt; free but unproven. Kill-test first.

## Mechanism
Score each image K times under mild perturbations/crops with existing models; use score VARIANCE as
signal (reals stable, fakes erratic — evidence unevenly distributed). Turns the contest's transforms
into our sensor. Output = feature for the ensemble.

## Pros / cons
+ zero training, reuses saved models, model-agnostic.
- K× inference cost; unproven strength; likely partial overlap with 01's patch-variance.

## Kill-test (30 min, saved models)
Compute score-variance over 8 views for 500 reals + 500 fakes with resnet_ft; report AUROC of the
variance alone. <0.6 → drop.

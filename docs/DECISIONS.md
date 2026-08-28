# Decisions

One line per decision. Newest first.

| Date | Decision | Why |
|---|---|---|
| 2026-08-26 | Approaches decoupled as `src/approaches/<name>/` folders on main; no experiment branches | Isolation without merge pain for 4 parallel workers (Thinh's call) |
| 2026-08-26 | Org `wheres-my-perry`, repo public | TechJam rules require public repo |
| 2026-08-26 | Progress tracked in `docs/PROGRESS.md`, decisions here | Keep docs brief and in one place |

## 2026-08-28 — Benchmark integrity protocol (Thinh)
Decision: benchmarks are validated model-free before use (scripts/shortcut_audit.py metadata
AUROC gate + scripts/size_audit.py), re-checked continuously, and no model result is reported
from an unaudited manifest. Trigger: official_val size confound (200x200 reals vs 1024+ fakes)
inflated all official numbers and manufactured three fake "miracles". Consequence: official_v2
(original-res COCO) replaces official_val; every future manifest passes the gate first.

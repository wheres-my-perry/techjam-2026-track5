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

## 2026-08-28 — Development moves onto the GPU server (Thinh)
Decision: Claude Code runs on the server (chim@157.66.47.161:2205, ~/techjam-2026-track5), which
becomes the primary working clone; commits and pushes happen there and the Mac clone drops to a
read-only mirror. Why: data, both 5090s and all Slurm logs live on the server, so the Mac->GitHub
->server hop added a round trip to every job check for no benefit, and ssh key auth from the Mac
was never established. Consequence: server needs a git identity + GitHub push credentials; the
old "server is a read-only mirror" rule in CLAUDE.md is reversed.

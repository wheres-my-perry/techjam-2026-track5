# Progress Log

Brief, current-state tracking. Newest entries first. Keep entries to 1–3 lines.

## Status: 🟡 Setup

**Now:** Eval harness built & smoke-tested; datasets scoped (docs/DATA.md).
**Next:** Decide approach (docs/IDEAS.md) → hardware check → baseline model. Also: invite teammates, webinar Aug 28 5pm SGT.

---

## Log

### 2026-08-26 (eval harness)
- Harness done: 15-condition transform grid, metrics (AUROC / bal.acc @ frozen thr / FPR@95TPR), `src/evaluate.py`, required `src/predict.py` CLI, 8 tests + synthetic smoke test passing. Usage: docs/EVAL.md.
- Dataset report: docs/DATA.md (WildFake = primary pool). Patch-scoring idea (Thinh): docs/IDEAS.md.

### 2026-08-26
- Repo created under org `wheres-my-perry` and seeded (README, docs, skills).
- Track 5 confirmed: **Robust Detection of AI-Generated Images Under Real-World Transformations**. Full brief in `docs/TRACK5_BRIEF.md` (key: <2B params, robustness grid, JSON prediction script required).
- 5 agent skills under `.claude/skills/` (aigc-detection, image-processing, ml-engineer, hackathon-shipping, demo-pitch).
- Contest facts (Devpost): 72-hr build Aug 29–Sep 1, submission Sep 1 12:00pm SGT, need public repo + 3-min YouTube demo.

<!-- Template:
### YYYY-MM-DD
- What changed / decided / shipped.
-->

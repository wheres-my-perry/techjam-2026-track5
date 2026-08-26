---
name: hackathon-shipping
description: Discipline for shipping a winning project inside a 72-hour hackathon — scoping, prioritization, team coordination for 4 people, and aligning work with TechJam judging criteria. Use when planning features, deciding what to cut, dividing work, or when the project is drifting from a demoable state.
---

# Hackathon Shipping (TechJam 2026)

Optimize for the judging rubric, in this order of leverage:

1. **Working end-to-end demo** — a thin slice that works beats a deep slice that doesn't.
2. **Problem insight** — the writeup must show we understood the track's real pain point.
3. **Presentation** — 3-min video and README quality are scored; budget real time for them.

## Rules

- **Demo-first development**: from hour ~6 onward there must ALWAYS be a runnable end-to-end demo, however fake the internals. Improve parts behind a stable interface.
- **Walking skeleton before features**: input → processing → output wired with stubs first.
- Every feature idea gets one question: *"does this change what judges see in 3 minutes?"* If no → backlog.
- Cut scope at the **feature** level, never by shipping half-broken features.
- Hard checkpoints (submission Sep 1, 12:00pm SGT):
  - T-24h: feature freeze. Only polish, fixes, video, writeup after this.
  - T-12h: record demo video (record early; re-record only if time allows).
  - T-3h: submission dry run — repo public, README complete, video uploaded and playable in incognito.

## 4-person split (default; adjust to strengths)

| Role | Owns |
|---|---|
| ML/CV lead | Model choice, `src/model.py`, `src/infer.py`, eval |
| Backend/integration | API, pipeline glue, deployment of demo |
| Frontend/demo | UI, demo flow, visualizations |
| Product/present | Problem framing, writeup, video, PROGRESS/DECISIONS docs, testing as a user |

Everyone commits to `main` behind small PRs; no branch lives > half a day.

## Team hygiene

- Update `docs/PROGRESS.md` at least every work session (1–3 lines).
- Log every non-obvious choice in `docs/DECISIONS.md` — this becomes the writeup for free.
- Merge conflicts and integration pain are the #1 hackathon time sink: integrate every few hours, not at the end.

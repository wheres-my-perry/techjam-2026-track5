# AGENTS.md — standing orders from Thinh (team lead). Re-read before every work session.

This file is the accumulated set of instructions Thinh has given, with the reason each one exists.
It is not background reading: **§1 is a procedure to execute, not advice to consider.**
Project rules live in CLAUDE.md; per-approach conventions in .claude/skills/project-conventions/.

---

## 0. RULE ZERO — EVERYTHING THINH SAYS GOES INTO HARD MEMORY, IMMEDIATELY
> "you also fucking write in hard memory that whatever I said you write in hard memory"
> "I don't want to say that observation a second time, don't treat my words lightly; this is
> compacted from 3 days of suffering"

When Thinh states an observation, a rule, a correction or a preference:

1. **Write it down BEFORE acting on it.** Not after the task, not "later when I document" — first.
   An instruction that only lives in the chat is lost at the next context boundary, and he then has
   to spend his time repeating what already cost him days to learn.
2. **Write it in all three places:**
   - `AGENTS.md` (this file) — the standing order, with the reason and the measurement behind it;
   - `CLAUDE.md` — if it is binding on every session, as a rule;
   - the session memory directory — so it survives beyond this repo checkout.
   If it constrains the data or the training, also encode it in `configs/canon6.yaml` and add a
   test in `tests/test_corpus_config.py`, because a rule in prose gets skipped and a rule in a gate
   does not.
3. **Record the REASON, not just the instruction.** "Equal buckets" is forgettable; "canonicalize
   only rescales images above 320, so unequal buckets train one forensic regime" is not.
4. **If you disagree, say so and argue** — he has explicitly invited that ("if not then fight back
   and it's perfectly fine to do so"). Silent non-compliance is the only wrong answer.
5. Never make him say it twice. If he repeats himself, that is a process failure, not a reminder.

---

## 1. DATA IS PRIORITY ONE

> "the data is top 1 priority, it has to be clean and flawless; much much more important than
> anything else: model, observation, etc"

A model trained on flawed data produces numbers that measure the flaw. Every headline this project
has lost was lost to data, never to modelling.

### 1.1 After EVERY change to the training set, look at the distribution again — yourself
> "have a look at data by yourself every time you modify training dataset"

Not "run the gates and read PASS". Do all of it:
```
python -m scripts.audit_all   --prefix data/manifests/<name>
python -m scripts.corpus_audit --prefix data/manifests/<name> --write-drop <drop.txt>
python -m scripts.content_audit --manifests <name>_train.csv <name>_val.csv
python -m pytest tests/test_corpus_config.py -q
```
then **actually look at the images**: build montages of random reals and fakes, and of each
suspect size bucket, and read them. Things found only by looking, never by a script:
- the fake class was flooded with bedrooms while the real class had none;
- the real class contains paintings while the fake class contains logos and product shots;
- `subject()` is a PATH REGEX, so its "other" bucket means "path matched no rule", not a content
  category — a 3.6:1 ratio there was a tagger artifact and would have sent me fetching 4 GB of
  the wrong data if I had trusted it.

### 1.2 EQUAL NUMBER OF IMAGES IN EVERY NATIVE-SIZE BUCKET
> "I want the buckets to have same numbers of images also; reason is that from different buckets we
> have to scale up and down and the model will receive photos with different characteristics"

Agreed, and the mechanism is sharper than the phrasing suggests. `canonicalize --long 320` only
shrinks images ABOVE 320, so:

| bucket | what happens to it before the model sees it |
|---|---|
| <=341     | essentially **NOT rescaled** — native pixels |
| 342-512   | shrunk ~1.1-1.6x |
| 513-768   | shrunk ~1.6-2.4x |
| 769-1024  | shrunk ~2.4-3.2x |

Downscaling attenuates high-frequency detail, and high-frequency structure is precisely what
separates generated images from photographs. So each bucket presents a DIFFERENT forensic regime.
canon6 was 65% <=341, meaning the model was mostly trained on un-rescaled thumbnails and learned
cues that are partly destroyed in every large image it will meet in deployment. This is the same
failure family as canon2 scoring 0/10 on wild photos.

**Rule: train and val hold the SAME number of images in every native-size bucket, per class.**
Set by `splits.equal_bucket` in configs/canon6.yaml and enforced by tests/test_corpus_config.py.
The binding constraint is always the FAKE side — measured pool: <=341 40,920 · 342-512 5,288 ·
513-768 5,272 · 769-1024 11,185, so the middle buckets decide the corpus size and are the ones to
grow (they are fed by ELSA alone, of which we had pulled 8 of 5,239 shards).

**Known consequence to state, not hide:** nothing we have generates at 342-768 except the SD family,
so those buckets are ~100% sd14/sd21/sdxl. "Mid-resolution" and "SD-family" are therefore partly
confounded in this corpus.

### 1.3 One-sided content: TEST ONLY, NEVER TRAIN
> "dirty data like fully bedroom and church and so on can be retained for testing, but definitely
> not for training"
> "if you train on bedroom only, the model would excel at classifying bedroom photos, and NOTHING
> ELSE; your choice"

Two independent harms, and the second is the one people forget:
1. **shortcut** — "bedroom => fake" is learnable without looking at whether the image is generated;
2. **competence** — a model whose fake class is one-third bedrooms learns bedroom detection and has
   nothing for a phone photo of a person. This is why canon2 scored **0/10** on real phone photos.

Measured: `ddim` was 19,093 rows (30% of the fake class) and its content was bedroom 76.4% /
church 23.6% — 100% in two subjects. It is now held out (`configs/canon6.yaml`).
**Balancing is not a substitute for removal, and removal has a mirror image:** adding real bedrooms
only reached 2.14:1, and removing ddim while keeping those reals flipped it to 12.55:1 the other
way. Both sides of a one-sided axis must move together.

### 1.4 Know what each gate CANNOT see
A gate that passes may be blind rather than clean. `shortcut_audit` and `size_audit` read the
CANONICAL files, which are all 176x176 after canonicalization, so they see constant
width/height/format and are structurally incapable of seeing NATIVE size. `canon_unseen6` passed
`shortcut_audit` at 0.617 while three of five native-size buckets contained no fakes at all and
reals ran to 7712px against fakes capped at 1024. Always ask: *what can this check physically not
see?* — then check that separately.

### 1.5 Audit it yourself; do not take a green tick as truth
> "I said you audit the dataset, and by that I don't mean just run the scripts, please check by
> yourself also"

---

## 2. READ THE PAST FINDINGS — THEY COST THREE DAYS
> "I gave you all the docs about the mistakes and so on, don't take them lightly, read all of them"
> "I suffered 3 days to experience all of the bug and flaws from the data; not for you to go
> through the same thing and waste me time"

Read before touching data: `docs/LESSONS_FOR_TEAMMATES.md` (all six sections),
`docs/DATA_AUDIT_2026-08-30.md`, `docs/DATA_AUDIT_2026-08-31.md`, `docs/DATA_STATUS_*.md`.
The findings are documented **and they work** — the ArtiFact `target`-label rule and the COCO
val2017 leak were both caught in the canon6 rebuild *because* they were written down.

## 3. WHEN YOU MISS SOMETHING, MAKE IT UNMISSABLE
> "everything I tell you that you missed you should write script or doc or whatever to retain the
> valuable lesson"

A lesson in prose gets skipped; a lesson in a gate does not. `content_audit.py` existed, caught the
bedroom bug once, and was **named nowhere in CLAUDE.md** — so it was skipped and the bug came back
at 92.7:1. Fixed by binding all seven gates in CLAUDE.md and adding `scripts/audit_all.py`.
Process lessons: `docs/LESSONS_FOR_TEAMMATES.md` section 6.

## 4. CONFIG + AUTOMATED TESTS
> "you keep everything involved in training in a config file, then build automated testings, like
> checking distribution"

`configs/canon6.yaml` is the single source of truth: canonicalization, routing, caps, splits,
hyperparameters, forbidden benchmark slices, invariants, gate list.
`tests/test_corpus_config.py` asserts the BUILT manifests satisfy it. Change the config, rebuild,
run the tests. **Never edit the rules inside the scripts.**

## 5. THE BRIEF IS THE SOURCE OF TRUTH — KEEP IT VERBATIM
> "drop the fucking brief, or keep it but also keep a clean version for reference, the single
> source of truth"

`docs/TRACK5_BRIEF_ORIGINAL.md` is the verbatim text: never edit, never annotate.
`docs/TRACK5_BRIEF.md` is our interpretation and is labelled as such. The condensation had hardened
an inference into a quotation ("the source settles that stacking is in scope" — it says no such
thing). Any doc that paraphrases a source keeps the source beside it.

### 5.1 "A subset of the augmentations" bounds WHICH, not HOW MANY
> "subset means that it could be many augmentations stacked on top of each other"

Training uses `--stack-aug 0.4 --stack-max 6`; evaluation reports stack depths 2..6 alongside the
brief's 15 single transforms. **The last run's config is not "the recipe":** canon5 shipped
`--stack-aug 0`, which was the baseline arm, not the intended one.

## 6. EVALUATION
- The unseen-generator set is the **overfit checker** — not a headline. Keep it in two groups that
  are never pooled: *unseen architecture* vs *unseen version of a family we train on*.
- Deduplicate any benchmark against training before quoting it (18% of one "unseen" real source
  was images we trained on).
- Evaluation sets with one-class size buckets are reported **size-matched only**.
- A number is quotable only if its data can be rebuilt from this repo. canon4's 0.9955 cannot be,
  and is therefore permanently unverifiable.
- Never report a number from a manifest that failed a gate. >=0.99 starts a shortcut hunt.

## 7. WORKING WITH THINH
- **Do not lie, do not overclaim.** Say plainly what was verified and what was not.
- ONE topic at a time. Correct him when he is wrong. Explain simply when asked — his words:
  "my english is bad", so use short sentences and concrete numbers, not jargon.
- Scope: no out-of-scope improvement work. Deadline pressure is real.
  > "focus on things that prepares for it ... as we may not be able to do it later as it would be
  > too late"
  Do the things that become IMPOSSIBLE later first: anything needing the GPU, the data, or the
  ephemeral box. Documentation is text and can be written any time — do it after.
- Get weights and artefacts OFF the ephemeral box as soon as they exist.

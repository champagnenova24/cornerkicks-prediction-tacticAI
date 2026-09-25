# Football set-piece analysis on StatsBomb open data

Two connected pieces of work over the StatsBomb open dataset:

| Directory | What it is |
|---|---|
| [`pipeline/`](pipeline/) | Possession/decision pipeline over all 4,235 matches, plus ball-trajectory reconstruction and a positivity audit for causal estimands. Documented below. |
| [`tacticai_task2/`](tacticai_task2/) | Corner kicks: a reproduction of Task 2 of TacticAI (shot prediction from corners), a set-diffusion generator for corner setups, and a delivery-target analysis over all 41,178 corners in the dataset. See [its README](tacticai_task2/README.md). |

Headline results from the corner work, with the caveats that belong to them,
are in `tacticai_task2/README.md`. The short version: the near post takes 39%
of all long corners and is measurably the worst target; inswinging vs
outswinging changes shot attempts but not expected goals; and several earlier
findings in that README are marked as superseded where a larger sample or a
bias correction overturned them.

## Data

This repository contains **code and derived results only**. It expects the raw
StatsBomb open data locally; the simplest placement is inside the repository
(it is gitignored):

```bash
git clone https://github.com/statsbomb/open-data.git open-data-master
```

`pipeline/common.py` resolves the dataset from `$OPEN_DATA`, then
`<repo>/open-data-master`, then `~/Desktop/open-data-master`, then
`~/Downloads/open-data-master`.

Stage 1 needs a `match_index.csv` at the root of the checkout, which is not
part of the StatsBomb repository. Build it once with:

```bash
cd pipeline && ../.venv/bin/python build_match_index.py
```

The intermediate parquets in `data/processed/` (719 MB) are **not committed** —
two of them exceed GitHub's 100 MB file limit. Drop a shared copy into
`data/processed/` (also gitignored), or regenerate them with:

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cd pipeline && ../.venv/bin/python stage1_matches.py && ../.venv/bin/python stage234_possessions.py && ../.venv/bin/python stage5_decision_points.py
```

## Attribution

Built on [StatsBomb open data](https://github.com/statsbomb/open-data), which is
free for research and non-commercial use with attribution. Any analysis or
insight derived from it — including everything in this repository — should
credit **StatsBomb** as the data source. See StatsBomb's own LICENSE and
[media pack](https://statsbomb.com/media-pack/) for their terms and logo.

Player height and weight in `tacticai_task2/player_bio.parquet` come from
[Wikidata](https://www.wikidata.org/) (properties P2048 and P2067).

## Pipeline

Run from `pipeline/` with the project venv (`.venv/bin/python`):

| Stage | Script | Output (`data/processed/`) |
|---|---|---|
| 1. Corpus split | `stage1_matches.py` | `matches.parquet` — 4,235 matches, `corpus` (pretrain vs causal_core = has 360), `split` (train/val/test by match hash, Women's World Cup 2023 = `ood_test`) |
| 2–4. Possessions, canonical frame, decision cleaning | `stage234_possessions.py` | `possessions.parquet` — one row per possession with terminal outcome (goal/shot/ball_lost/out/foul_won/…), goal flag, xG sum. `decisions.parquet` — one row per on-ball decision (pass/cross/carry/dribble/shot) by the possession team, coordinates in the canonical attacking frame (goal at x=120), Δt clipped at 10 s, action target discretized to a 12×8 zone grid, outcome-forward labels (`xg_after`, `goal_after`, `n_remaining`) |
| 5. (state, action, outcome) triples | `stage5_decision_points.py` | `decision_points.parquet` — decisions in the 426 matches with 360 coverage, joined to freeze frames, off-camera excluded. `freeze_frames.parquet` — one row per visible player per decision (`teammate`/`actor`/`keeper` flags). `visible_area.parquet` — camera polygon per decision (the missingness mask) |
| 6. Tensors | `tensorize.py` | `tensors/{train,val,test,ood_test}.npz` — fixed-shape arrays for a set encoder: up to 22 players × (x, y, teammate, actor, keeper) with a slot mask, the camera polygon rasterized to a 16×24 grid of 5-yd cells (`visible`, the attention mask for "unknown" vs "empty"), the action, the next 16 decisions of the possession padded with a mask and a `truncated` flag, and the outcome labels. See below |
| Checks | `validate.py` | prints corpus stats, invariant assertions, split tables, positivity preview |
| Ball trajectory (optional) | `ball_trajectory.py` | `corner_trajectories.parquet` — reconstructed ball path per corner delivery, 60 sampled points each (`t`, `x`, `y`, `z`), joinable to `decision_points.parquet` on `uuid` |
| Figures | `viz_corners.py <outdir>` | `corner_trajectories.png`, `corner_trajectories_by_technique.png` |

### Supporting modules and analyses

These write no parquet — they are libraries, one-off setup, or scripts that
print a report to stdout. All are current.

| Script | What it is |
|---|---|
| `visibility.py` | Library. The camera polygon as a missingness mask: rasterize to the 5-yd grid, and the share of any rectangle or disc the camera saw, measured over its on-pitch part (the polygons stop at the touchlines). Used by `tensorize.py`, `positivity.py` and `tacticai_task2/defense_features.py`. Imported, not run. |
| `common.py` | Library. Shared constants and helpers: dataset path resolution (`$OPEN_DATA` → repo root → Desktop → Downloads), the canonical frame flip, the 12×8 zone grid, the deterministic match split. Imported, not run. |
| `build_match_index.py` | One-off setup. Builds `match_index.csv` at the root of the open-data checkout, which stage 1 expects and StatsBomb does not ship. Run once before `stage1_matches.py`. |
| `eval_trajectories.py` | Validates the reconstructed corner trajectories by the gap/fidelity method. **Check 0 is the regression guard on the swing convention** — an inswinger must bow away from goal, and getting it backwards puts the ball out of play behind the goal line. Prints a report. |
| `positivity.py` | Positivity / overlap audit. `E[Y | do(A=a), S=s]` is identified only where `P(A=a | S=s) > 0`; a conditional generative model will happily extrapolate into states where an action is never taken, so the violation has to be measured before any causal claim, not discovered in the output. Reports support per (state bin, action), empirical propensities, the cost of refining the action space, and the shoot-vs-pass contrast by pitch region. Prints a report. |
| `estimand_box.py` | The `do(shoot)` vs `do(pass)` estimand restricted to the penalty area, end to end: restrict the population, build confounders from the 360 freeze frames, fit `P(shoot \| S)`, check overlap, trim to common support, then contrast naive vs IPW vs propensity-stratified effects with a match-clustered bootstrap. Follows directly from `positivity.py`, which showed the full-pitch contrast is dominated by states where nobody ever shoots. Prints a report. |

The same adjustment machinery, applied to corners rather than to shoot-vs-pass,
is `tacticai_task2/technique_effect.py` — a separate estimand, not a
replacement.

## Ball trajectory reconstruction

`ball_trajectory.py` is a standalone module — nothing in stages 1–5 depends on
it. Importable as `reconstruct(start, end, duration, height_class, technique)`
or `reconstruct_event(raw_pass_event)`; run as a script it writes the corner
parquet.

It has one external consumer: `tacticai_task2/synthetic_trajectories.py` imports
`reconstruct` to put ball paths on diffusion-generated corners, so this module
is the single source of truth for the physics across both projects. That
consumer reflects endpoints into the tacticai frame (corners mirrored to a kick
at (120, 0)) before calling in — valid because the reflection fixes the goal
centre (120, 40), leaving the swing orientation unchanged.

**Observed vs assumed.** StatsBomb records the ball only at event boundaries.
Observed per corner: start `location`, `pass.end_location`, `duration` (flight
time), `pass.height` (Ground/Low/High) and `pass.technique` (swing direction).
Reconstructed: the path between the endpoints — vertical profile is genuine
ballistics driven by the measured flight time, capped below shoulder height for
Low and zero for Ground; lateral bow takes its **sign** from `technique` but its
**magnitude** (`swing_yards`, default 2.5) is a free prior. The data does not
observe curvature magnitude; do not fit claims to it.

**Contact height.** The flight is solved to arrive at the height the ball was
met — `z(0) = 0`, `z(T) = contact height` — not to land back on the turf. The
body part of the next touch is observed for ~93% of corners (55% headers), and
`CONTACT_HEIGHT_YD` maps it to a height (Head 2.2 yd, Keeper Arm 2.6, foot 0.3,
chest/other 1.2). The *category* is observed; the height per category is a
prior — tune it, don't fit to it. Assuming every corner drops to the ground
biases the apex low by ~30% on a typical headed delivery; over all High Pass
corners the corrected mean apex is 5.89 yd vs 5.02 yd under that assumption.
Passing `end_height=0` recovers the old symmetric apex = gT²/8 exactly.

**Swing convention.** An inswinger's arc bulges *away* from the goal. With both
endpoints fixed, an arc bulging to one side arrives with its velocity rotated
toward the other, so a delivery that arrives moving goalward must bulge away
from goal. Getting this backwards is not just cosmetic: a corner starts *on*
the goal line, so a goalward bow puts the ball out of play behind it. The
amplitude is additionally clamped to keep every sampled point inside the pitch;
this binds on ~65% of outswingers (whose arc hugs the goal line) and ~1% of
inswingers. `eval_trajectories.py` check 0 is the regression guard.

## Cleaning decisions baked into stages 2–4

- Events sorted by `index` (never timestamp — 0.004% of raw timestamps are
  out of order).
- Derived/mirror events dropped: Ball Receipt, opponent Pressure/Block/
  Goal Keeper annotations. Their signal survives as `under_pressure` and in
  the possession terminal label.
- Events by the non-possessing team are rotated 180° into the possession
  team's attacking frame before any coordinate is used.
- `dt` clipped at 10 s (~p97 of inter-event gaps); raw value kept in `dt_raw`.
- `off_camera` events kept in `decisions.parquet` (flagged) but excluded from
  `decision_points.parquet`.

## Stage 6: tensors, and the positivity audit by state bins

Both items that used to sit here as "not yet built" now exist.

**Tensorizer** (`tensorize.py`, ~1 min). 683,133 decision points in 418 matches:

| split | N | matches | file |
|---|---:|---:|---|
| train | 462,899 | 276 | `tensors/train.npz` (94 MB) |
| val | 62,631 | 38 | `tensors/val.npz` |
| test | 67,154 | 40 | `tensors/test.npz` |
| ood_test | 90,449 | 64 | `tensors/ood_test.npz` (Women's World Cup 2023) |

- **Missingness mask.** A median of 16 players is visible and the camera sees
  a median 28% of the pitch. A set encoder over the players alone reads every
  unseen region as empty, so the polygon ships as a cell mask: add one token
  per 5-yd cell and mask the cells the camera did not see, and "empty and seen"
  and "unknown" become different inputs. Sanity check that the frames agree:
  97.8% of visible players sit in a cell marked visible (the rest are on the
  polygon edge at 5-yd resolution), and the ball's own cell is visible for 99.3%.
- **Horizon.** Possessions are long-tailed (median 7 decisions remain after a
  decision point, p90 28), so no fixed window covers them. H = 16 holds the whole
  remainder for 75.8% (H = 8 would hold 55%), and `truncated` marks the rest —
  so a sequence model can tell "the possession ended" from "the window ended".
  The outcome labels always cover the whole possession. The continuation comes
  from `decisions.parquet` and so includes off-camera steps, which happened even
  though they have no freeze frame.

**Positivity by state bins** (`positivity.py`). Sections 1–4 were already the
audit over `ball zone × under_pressure` (192 bins); section 5 refines the state
with the freeze frame — opponents within 5 yd of the ball: 0 / 1 / 2+ — and uses
the polygon to decide when that count means anything. A count of 2+ is known
regardless; a 0 or 1 is only trusted when the camera saw ≥ 90% of the disc,
otherwise the state is **unobserved**:

- 4.1% of the raw "0 opponents nearby" decisions are actually unobserved — a
  freeze-frame-only encoder would call them unpressured.
- Refining 192 → 767 bins costs shots the most: the share of decisions sitting
  in a bin where shooting has never been observed rises from 62.0% to 74.5%.
  Pass and carry stay supported everywhere; cross and dribble were already
  unsupported in ~80% of states.
- The `estimand_box.py` population (penalty area, shoot vs pass) keeps
  positivity under every local level — P(shoot) is 0.67 / 0.46 / 0.58 with
  0 / 1 / 2+ opponents near the ball — and only 0.8% of it (92 decisions) is
  unobserved.
- **It also contains 248 penalties** (2.0%; a shot from (108, 40) with play
  pattern "Other", xG exactly 0.784). Passing is not an option from the spot,
  so they are a structural positivity violation of the same kind as shooting
  from the own half, and they are what makes the "unobserved" row look so
  valuable (P(shoot) 0.87, mean xG 0.63): 73 of those 92 are penalties, whose
  freeze frames have a narrow camera. They also inflate the "0 opponents"
  row. `estimand_box.py` currently keeps them; they should be dropped before
  its do(shoot) vs do(pass) contrast is quoted. This is exactly the kind of
  thing the audit exists to find, and it is flagged rather than silently fixed
  here because it changes a published estimand.

**Upstream data note.** StatsBomb's current `three-sixty/3845506.json` is
corrupt at the source (stage 5 skips it and logs it to `stage5_skipped.txt`), so
a rebuild from today's open-data has 425 usable 360 files, not 426. See
`tacticai_task2/README.md` → *Things that can mislead you* #7 for what that does
to the corner files and how `defense_features.py` avoids it.

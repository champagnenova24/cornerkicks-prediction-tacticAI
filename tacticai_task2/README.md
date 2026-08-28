# TacticAI Task 2 (shot prediction from corners) on StatsBomb open data

> ## Which scripts are current
>
> The repository keeps superseded scripts for provenance — every result below is
> reproducible from the script that produced it, including the ones later
> overturned. **Two naming traps are worth flagging first:**
>
> - **`extract_corners.py` and `extract_corners_full.py` are not versions of each
>   other.** Both are current. The first builds the 2,473-corner **360 subset**
>   with freeze frames, receiver matching and velocities — everything the models
>   need. The second identifies all **41,178 corners** across 4,235 matches, with
>   no freeze frames. Wider but shallower; neither replaces the other. Same for
>   `add_delivery.py` (subset) vs `extract_delivery_full.py` (all matches).
> - **`tactic_suggest3.py` is NOT the live recommender — `tactic_recommend.py`
>   is.** But `tactic_suggest.py` *is* still live as an imported dependency
>   (`ddim_inpaint`, `draw_panel`), even though its own published numbers are
>   superseded. Do not delete it.
>
> ### Corner extraction and attributes
>
> | Script | Status | Scope |
> |---|---|---|
> | `extract_corners.py` | **current** | 360 subset, 2,473 corners: freeze frames, receiver, velocities |
> | `extract_corners_full.py` | **current** | All 4,235 matches, 41,178 corners, no freeze frames |
> | `add_delivery.py` | **current** | Delivery attributes, 360 subset |
> | `extract_delivery_full.py` | **current** | Delivery attributes, all matches |
> | `onpitch.py`, `fetch_bio.py` | **current** | On-pitch XI; Wikidata height/weight |
> | `paths.py` | **current, library** | Resolves the StatsBomb checkout (`$OPEN_DATA` → repo root → Desktop → Downloads). Imported, not run. Use it instead of hardcoding a data path |
>
> ### Models
>
> | Script | Status | Notes |
> |---|---|---|
> | `train_eval.py` | **current** | Receiver + shot models, Eq. 1, 9 configs × 5 seeds. `--check-cancel` runs the softmax-cancellation guard instead of training |
> | `diffusion_corners.py` | **current** | Set-diffusion model, training, DDIM sampler |
> | `diffusion_eval.py` | **current** | Realism discriminators + guidance check |
> | `quality_metrics.py`, `tstr_eval.py`, `gallery.py` | **current** | Extended metrics, TSTR, figures |
> | `ablate_batch.py`, `ablate_projection.py` | **current** | Ablation records (both negative results) |
> | `finish_ablation.py` | helper | One-off: evaluates the batch-512 checkpoint. Not an entry point |
>
> ### Recommendation — read this chain carefully
>
> | Script | Status | Why |
> |---|---|---|
> | `tactic_suggest.py` | **superseded results, live dependency** | v1. `tactic_recommend.py` imports `ddim_inpaint` and `draw_panel` from it. Its own best-of-6 numbers carry the selection bias |
> | `tactic_suggest2.py` | superseded | v2, best-of-k selection — the bias itself |
> | `tactic_suggest3.py` | superseded | v3. Maximised P(shot), which moves attempts but not xG; also single-model selection |
> | **`tactic_recommend.py`** | **CURRENT** | xG-based, cross-fitted selection, ball path attached. **This is the live recommender** |
>
> ### Delivery and defence analysis
>
> | Script | Status | Notes |
> |---|---|---|
> | **`delivery_target_full.py`** | **CURRENT** | Authoritative zone ranking, 41,178 corners |
> | `delivery_target.py` | **partly current** | Zone ranking superseded by the full set, but its defence-conditional and retention×payoff analyses need freeze frames and exist nowhere else |
> | `defense_features.py` | **current** | Defensive descriptors incl. the zonal-vs-man marking proxy |
> | `technique_effect.py` | **current** | Inswing vs outswing: IPW, cluster bootstrap, interaction scan |
> | `synthetic_trajectories.py` | **current** | Ball paths for synthetic corners + ball-consistency metric |
>
> ### Run order
>
> ```
> extract_corners.py -> onpitch.py -> fetch_bio.py -> add_delivery.py -> train_eval.py
>                    -> diffusion_corners.py -> diffusion_eval.py -> tactic_recommend.py
> extract_corners_full.py -> extract_delivery_full.py -> delivery_target_full.py
> defense_features.py -> technique_effect.py
> ```
>
> `extract_corners_full.py` requires `data/processed/decisions.parquet` from the
> pipeline (see the top-level README), which in turn needs `match_index.csv` —
> build it once with `pipeline/build_match_index.py`.
>
> Device selection is centralised in `diffusion_corners.pick_device()`
> (CUDA → MPS → CPU), so the torch scripts run on a cluster GPU or a Mac
> unchanged. Don't reintroduce a hardcoded `"mps"`.
>
> ### Where results are written
>
> | Directory | Holds | Rule |
> |---|---|---|
> | `results/` | every `.csv` a script writes | tables, metrics, splits, loss curves |
> | `figures/` | every `.png` | |
> | `models/` | every `.pt` | diffusion checkpoints |
> | *(script dir)* | `.py` and `.parquet` | derived data stays next to the code that reads it |
>
> Parquet is deliberately exempt: it is the interchange format between scripts,
> not a result to be read by a human, and moving it would have meant rewriting
> every load path in the project for no gain.
>
> **`check_paths.py` enforces this statically** — it walks the AST of every
> script and flags any `to_csv` / `savefig` / `torch.save` whose path does not
> resolve under one of the three directories:
>
> ```bash
> python check_paths.py --strict   # exit 1 if anything is unmigrated
> ```
>
> This exists because the failure it catches is silent. A write left pointing
> at the old location still *succeeds* — it creates the file next to the
> source, nothing raises, the script prints its summary, and the copy under
> `results/` quietly goes stale. "Did it run?" cannot detect that. Pair it with
> the runtime check, after running a script:
>
> ```bash
> git status --porcelain tacticai_task2/ | grep -E '^\?\? tacticai_task2/[^/]+\.(csv|png|pt)$'
> ```
>
> Any result file showing up as untracked in the script directory is a write
> that was not migrated. `outdirs.py` creates the three directories on import,
> so a fresh checkout that has deleted one does not fail after doing the work.
>
> Filenames in the prose below are written bare (`quality_metrics.png`); the
> directory follows from the rule above. The tables carry the full path.

Reproduction of Task 2 of *TacticAI: an AI assistant for football tactics*
(Wang et al., Nat. Commun. 15:1906, 2024), Eq. (1):

    P(shot|corner) = sum_i P(shot | receiver = i, corner) * P(receiver = i | corner)

using StatsBomb open data (`~/Desktop/open-data-master`) instead of Liverpool FC
tracking data. Player snapshots come from StatsBomb 360 freeze frames (anonymous
visible players with teammate/actor/keeper flags) at the moment the corner is taken.

## Task construction (mirrors the paper)

- **Corners**: every `Pass` event with `pass.type = Corner` in the 426 matches
  with 360 coverage that has a freeze frame -> 2,473 corners (2,425 after
  requiring >= 5 visible players per team).
- **Receiver** = first player to touch the ball after the corner (ball receipt,
  clearance, keeper claim/punch, interception, ...), stopping at restarts.
  Matched to the nearest freeze-frame player with the correct team flag
  (median match distance 2.6 m; matches > 10 m excluded from training).
- **Shot label** (paper's definition: first attacking-team touch is a shot
  attempt): the first attacking touch is a `Shot`, or the corner pass carries
  `shot_assist`/`goal_assist` (first-time shots recorded as receipt + shot).
  Shot rate 20.2% (paper: 24.2%; 1736/7176).
- All corners reflected so the kick is taken from (120, 0) — the analog of the
  paper's D2 symmetry handling.
- Protocol: random 80:20 corner split (paper's protocol), 10% of corners as
  validation for F1 threshold tuning, 5 seeds, F1 on the shot class + ROC-AUC
  (threshold-free ranking check).

## Velocity estimation from consecutive 360 frames

StatsBomb 360 has no tracking, so per-player velocity is estimated from the
pair of freeze frames that bracket the ball flight:

1. Frame A = corner event frame (at the kick). Frame B = the next event that
   carries a frame (the first touch: receipt/clearance/keeper), accepted when
   0.3 s <= dt <= 6 s. Found for 79% of corners; median dt = 1.58 s.
2. Frame B is rotated into frame A's orientation when the touching event
   belongs to the defending team (coordinates and teammate flags flip).
3. Anonymous players are matched with a Hungarian assignment (scipy
   `linear_sum_assignment`) on pairwise distance, separately per team side and
   keeper flag; matches farther than 9.5 m/s * dt + 2 m are rejected.
4. velocity = displacement / dt, speed capped at 9.5 m/s. Unmatched players
   get v = 0 and `has_vel = 0`.

Sanity: median matched speed 2.2 m/s, p90 = 5.0 m/s, only 5% above 6 m/s.

**Caveat**: these are ball-flight *average* velocities, i.e. they include some
post-kick reaction to the delivery. TacticAI's velocities are pre-kick tracking
estimates. Results with velocity features are therefore mildly optimistic
versus a true pre-kick system (there is no pre-kick frame pair in this data:
only 2% of corners have another frame within 10 s before the kick).

## Player height/weight (Wikidata)

`fetch_bio.py` looks up all 4,079 players appearing in the 426 matches on the
Wikidata API (nickname first, then full legal name; candidate entity accepted
when its description mentions football/soccer) and pulls height (P2048) and
mass (P2067). Coverage: QID found for 89% of players, height for 81%, weight
for 39% (heights recorded in metres/feet are unit-repaired; outliers dropped;
missing values imputed with per-gender means). Freeze-frame dots are
anonymous, so individual bio can only be attached to identifiable roles; it
enters the models as:

- global aggregates: attacking/defending team mean and max height, mean
  weight, defending GK height (on-pitch XI reconstructed per corner by
  `onpitch.py` from Starting XI / Substitution / Tactical Shift / red cards);
- receiver height/weight in the ground-truth-receiver *diagnostic* only
  (they cannot be used in Eq. 1 inference, where candidates are anonymous).

## Models (shallow analogs of the paper's GNNs)

- `P(receiver|corner)`: conditional logit over the variable-size set of
  visible candidates — ensemble of 3 small MLPs (16 tanh units) scoring each
  candidate, softmax across candidates; 16 geometric features per candidate
  (+5 velocity features in the `vel` configs).
- `P(shot|receiver, corner)`: histogram gradient boosting on global + receiver
  candidate features, trained with the ground-truth receiver (teacher forcing,
  as in the paper), marginalized over the receiver model at inference (Eq. 1).
- `P(shot|corner)` direct: same classifier family on global features only
  (incl. pooled per-player aggregates and the 3 attackers nearest goal).

## Results (mean +- std over 5 seeds)

### Delivery attributes from the corner pass event

`add_delivery.py` pulls `technique` (Inswinging / Outswinging / Straight),
`height` (High / Low / Ground Pass) and `body_part` off the corner pass event
-> `corner_delivery.parquet`. Marginal shot rates over all 2,473 corners:

| technique | rate | height | rate | body part | rate |
|---|---|---|---|---|---|
| Outswinging (670) | 29.0% | High Pass (2,088) | 23.2% | Right (1,504) | 20.6% |
| Straight (201) | 28.9% | Low Pass (65) | 12.3% | Left (969) | 19.5% |
| Inswinging (1,282) | 18.8% | Ground Pass (320) | 1.9% | | |
| Missing (320) | 1.9% | | | | |

`Missing` technique and `Ground Pass` are the **same 320 short corners** --
a subgroup that almost never yields a direct shot.

**Leakage boundary**: `technique` and `height` describe the delivery *after*
it is struck, so `vel+delivery` is a retrospective model, not a pre-kick
predictor. `body_part` is knowable in advance, so `vel+prekick` (foot only)
is the leakage-safe variant. Both are reported below.

> **Every number in this section was re-measured after a reproducibility fix.**
> The train/val/test id collections were Python `set`s, and iterating a set of
> strings depends on `PYTHONHASHSEED`, which varies per process. That order
> reached HistGradientBoosting's `early_stopping` split — which holds rows out
> by position — so the same seed produced a different model in every run. On
> `vel` seed 0, `auc_direct` ranged over 0.714–0.741 across four hash seeds,
> a spread wider than the differences between the configurations being
> compared, while the order-independent `recv_top3` never moved. The
> collections are now ordered arrays and four hash seeds agree to six
> decimals. Numbers published before this fix were one draw from that lottery
> and do not reproduce; the conclusions below survived it, with one exception
> flagged in finding 3.

| Metric                       | base (static) | + velocity    | + vel + bio   | Paper (tracking + GNN) |
|------------------------------|---------------|---------------|---------------|------------------------|
| Receiver top-1               | 0.20 +- 0.01  | 0.30 +- 0.02  | 0.30 +- 0.02  | n/a                    |
| Receiver top-3               | 0.41 +- 0.01  | 0.53 +- 0.01  | 0.53 +- 0.01  | 0.782 +- 0.039         |
| Direct shot F1               | 0.34 +- 0.02  | 0.43 +- 0.02  | 0.43 +- 0.02  | 0.52 +- 0.03           |
| **Eq. 1 decomposed shot F1** | 0.35 +- 0.02  | 0.44 +- 0.05  | 0.41 +- 0.10  | 0.68 +- 0.04           |
| Always-predict-shot F1       | 0.34 +- 0.02  | 0.34 +- 0.02  | 0.34 +- 0.02  | ~0.39 (implied)        |
| Direct shot AUC              | 0.55 +- 0.03  | 0.72 +- 0.02  | 0.71 +- 0.01  | n/a                    |
| Eq. 1 AUC                    | 0.58 +- 0.04  | 0.74 +- 0.01  | 0.74 +- 0.01  | n/a                    |
| Eq. 1 AUC, uniform receiver  | 0.54 +- 0.04  | 0.49 +- 0.05  | 0.49 +- 0.03  | n/a                    |
| Given true receiver: AUC     | 0.95 +- 0.01  | 0.95 +- 0.01  | 0.95 +- 0.01  | n/a                    |
| Given true receiver: F1      | 0.84 +- 0.03  | 0.83 +- 0.03  | 0.84 +- 0.02  | n/a                    |

(The receiver model ignores bio by construction — candidates are anonymous —
so recv_top1/top3 are identical between `vel` and `vel+bio`. A GT-receiver
diagnostic that additionally feeds the true receiver's own height/weight
reaches AUC 0.953 +- 0.010 vs 0.953 +- 0.008 without: no gain.)

Delivery configurations (5 seeds). Receiver top-3 is 0.531 for every config in
this first block, because there the receiver model uses per-candidate features
only; the `+recv*` block below is what changes that.

| config | recv top-3 | direct F1 | direct AUC | Eq. 1 F1 | Eq. 1 AUC |
|---|---|---|---|---|---|
| vel | 0.531 | 0.433 | 0.719 | 0.436 | 0.738 |
| vel+prekick (foot only) | 0.531 | 0.433 | 0.719 | 0.437 | 0.738 |
| **vel+delivery** | 0.531 | 0.455 | **0.757** | 0.432 | 0.750 |
| vel+bio+delivery | 0.531 | 0.458 | 0.754 | 0.445 | 0.750 |

Three findings:

1. **Delivery attributes are the biggest gain since velocities -- for the
   direct model**: AUC 0.719 -> 0.757, winning on 5/5 seeds paired
   (+0.031, +0.041, +0.042, +0.033, +0.043). Retrospective, per the leakage
   note above.
2. **Foot alone is worthless** (`vel+prekick` is identical to `vel` to three
   decimals). Footedness only matters through its interaction with the side
   of the pitch -- which is exactly what `technique` encodes directly, and
   what the left/right reflection erased.
3. **The decomposition's advantage disappears**: with velocities alone Eq. 1
   beat the direct model (0.738 vs 0.719, on 5/5 seeds); with delivery
   features the direct model overtakes it (0.757 vs 0.750, Eq. 1 winning on
   only 1/5 seeds). The reason is architectural -- the delivery features
   reach the *conditional* shot model but not the **receiver** model, whose
   candidate features are purely geometric, so marginalizing over an
   uninformed receiver distribution dilutes the global signal. The next
   section acts on that diagnosis.

   *(This is the one finding the reproducibility fix moved. The pre-fix gap
   was 0.761 vs 0.752; the direct model still wins, and now does so on 4/5
   seeds rather than on a single unreproducible draw. The direction held, the
   margin was never as precise as three decimals implied.)*

### Globals in the receiver model (TacticAI's `g`)

Finding 3 says the receiver model cannot see what the conditional model can.
`add_globals` fixes that by broadcasting the per-corner global vector onto
every candidate row — the analog of the graph feature vector `g` that
TacticAI's receiver head reads alongside its node features.

**How it can work at all.** The globals are constant across a corner's
candidates, so under a *linear* scorer they add the same term to every score
and cancel exactly in the softmax: appending them would be a no-op, and the
gradient on their weights is identically zero. They do work only through the
tanh hidden layer, where they shift the pre-activations and change how the
per-candidate geometry is weighed. Globals modulate the reading of the
geometry; they never vote for a candidate on their own.
`train_eval.py --check-cancel` asserts both halves of that (linear difference
2.6e-15, gradient 1.6e-17, hidden-layer difference 2.4e-01) so the mechanism
cannot silently disappear if someone sets `hidden=0`.

| config | globals → receiver | recv top-1 | recv top-3 | Eq. 1 F1 | Eq. 1 AUC |
|---|---|---|---|---|---|
| vel | none | 0.297 | 0.531 | 0.436 | 0.738 |
| vel+recvg | all 41 | 0.248 | 0.470 | 0.424 | 0.705 |
| vel+delivery | none | 0.297 | 0.531 | 0.432 | 0.750 |
| vel+delivery+recvg | all 49 | 0.293 | 0.519 | 0.431 | 0.736 |
| **vel+delivery+recvd** | **delivery only (8)** | **0.335** | **0.575** | **0.452** | **0.761** |

**The whole global vector hurts; the delivery block alone helps.** Handing the
receiver model all 41 base+velocity globals costs it 6 points of top-3
(0.531 → 0.470) and drags Eq. 1 AUC down to 0.705 — worse than doing nothing.
Those globals are mostly pooled restatements of the candidate features the
model already has (`n_att`, `mean_att_dg`, the three-nearest-attacker block),
so they roughly double the input width of a 16-unit layer trained on ~1,700
corners while adding no within-corner information. Restricted to the eight
delivery one-hots — genuinely new information, describing the *kick* rather
than any player — the same mechanism gains **+0.044 top-3 on 5/5 seeds**
(+0.041, +0.050, +0.054, +0.028, +0.046) and **+0.011 Eq. 1 AUC on 4/5**.

**What it does and does not settle.** `vel+delivery+recvd` has the best Eq. 1
AUC in the project (0.761) and erases the deficit to the direct model — but
erases is the right verb: Eq. 1 goes from −0.008 (winning 1/5 seeds) to +0.004
(2/5). The two are now indistinguishable, which is not the same as the
decomposition winning. On five seeds this margin cannot be called either way.

**Which attribute does the work — and a prediction that failed.** The obvious
guess is short corners: a ground pass goes to the near man, and a receiver
model that cannot see `height` would not know. That is wrong. Splitting the
holdout, Ground Pass corners were already at 0.952 top-3 before the fix —
geometry alone nearly saturates them, because the near man is also the
candidate nearest the corner arc. The gain is on *lofted* deliveries
(+0.045) rather than short ones (+0.032). Decomposing the block:

| globals given to the receiver model | cols | top-3 | vs none |
|---|---:|---:|---:|
| none | 0 | 0.531 | — |
| **technique only** | 4 | **0.573** | **+0.041** |
| height only | 3 | 0.546 | +0.014 |
| foot only | 1 | 0.529 | −0.002 |
| all delivery | 8 | 0.575 | +0.044 |

**Technique is essentially the whole effect.** Inswing versus outswing decides
which part of the box the ball is travelling toward, and therefore which
attackers are plausible receivers at all — information no snapshot of player
positions contains. Note that `foot only` adds nothing here either, the exact
echo of finding 2 on the shot side: footedness matters only through the swing
it produces, and `technique` already encodes that directly.

The leakage boundary applies unchanged — `technique` is known only after the
ball is struck, so `+recvd` is a retrospective model. There is no pre-kick
version of this gain, since foot, the one pre-kick attribute, is worth zero.

Takeaways:

1. The paper's core premise is strongly confirmed: conditioned on the true
   receiver, the shot outcome is highly predictable (AUC 0.95) — receiver
   identity carries most of the signal, which is why Eq. 1 helps.
2. The decomposition's *realized* advantage scales with receiver-predictor
   quality, and shows up in ranking rather than in thresholded F1: over the
   direct model, Eq. 1 gains +0.025 AUC with static features (on 3/5 seeds)
   and +0.019 with velocities (on 5/5), while the F1 gap stays inside noise
   (+0.005 and +0.003). Ranking is the honest place to read it — F1 depends
   on a tuned threshold and its seed-to-seed spread (±0.05) swamps the effect.
3. Velocities are the single biggest data upgrade: direct AUC 0.55 -> 0.72,
   Eq. 1 AUC 0.58 -> 0.74. This mirrors the paper's emphasis on movement:
   who is running where at the kick is most of the tactical signal
   (with the flight-window caveat above).
4. Weighting candidates by the learned receiver distribution matters: with
   velocity features, uniform averaging collapses to chance (AUC ~0.49) while
   receiver-weighted marginalization reaches 0.74.
5. Height/weight adds nothing here (vel -> vel+bio flat everywhere, and the
   true receiver's own bio adds no AUC in the diagnostic). Plausible reasons:
   anonymity forces bio down to team aggregates that are near-constant per
   team; the paper could attach height/weight to every tracked player and
   still treated them as a minor feature.
6. Giving the receiver model the delivery attributes is the largest gain it
   has had since velocities: top-3 0.531 -> 0.575 on 5/5 seeds, and it is
   what closes Eq. 1's deficit to the direct model. But feeding it the *whole*
   global vector makes it worse — the useful part is the ~8 columns that
   describe the kick rather than restating the players.
7. Remaining gap to the paper: pre-kick velocities vs flight-window
   velocities, full 22-player tracking vs ~75% visibility, exact receiver
   labels vs nearest-player matching, 7,176 vs 2,425 corners, geometric deep
   learning vs shallow models, per-player height/weight vs aggregates. Note
   receiver top-3 is now 0.575 against the paper's 0.782 — the remaining gap
   is narrower than it was, and is where the decomposition's value lives.

## Synthetic corner generation with diffusion (paper Task 3 analog)

`diffusion_corners.py` trains a conditional set-diffusion model that generates
per-player (x, y, vx, vy) for a corner setup:

- roster template (teammate / keeper / taker / velocity-observed flags per
  slot) taken from a real corner — the analog of TacticAI adjusting a fixed
  set of players;
- outcome conditioning (shot / no shot) with classifier-free guidance
  (label dropout 15%); guidance weight w = 1 for realism sampling, w = 2 for
  pushing setups toward/away from shot-producing configurations;
- denoiser: permutation-equivariant transformer over player slots (1.09M
  params, d=160, 5 layers, no positional encoding), DDPM cosine schedule,
  epsilon prediction, EMA, DDIM sampling (200 steps), position-jitter
  augmentation; velocity loss masked to observed velocities;
- trained on 1,940 corners, 485 held out (`diffusion_split.csv`).

**Laptop verdict (the go/no-go gate you asked for)**: an Apple M3 Pro handles
this easily — the script measures throughput before committing (aborts over a
90-min budget). Measured: 42 steps/s for a 0.56M model (30k steps = 12 min),
23 steps/s for the 1.09M model (60k steps = 44 min). Sampling 3 x 485 corners
at 200 DDIM steps takes ~2 min. This is set-diffusion over ~22 tokens, not
image diffusion; more compute is only needed for hyperparameter sweeps or
full trajectory diffusion.

### Results (`diffusion_eval.py`, held-out templates)

Realism — can a classifier tell real from synthetic? (paper's CVAE: MLP
F1 = 0.53 +- 0.05, i.e. indistinguishable; 0.5 = chance, higher = more
detectable):

| Discriminator                    | 0.56M / 30k steps | 1.09M / 60k steps |
|----------------------------------|-------------------|-------------------|
| MLP F1 (paper's test)            | 0.63              | 0.62              |
| MLP AUC                          | 0.69              | 0.70              |
| HGB F1 / AUC (stronger check)    | 0.61 / 0.69       | 0.69 / 0.74       |

(Sampling the realism set with w = 2 instead of w = 1 worsens MLP F1 to 0.68 —
guidance deliberately moves samples off-manifold, so realism must be evaluated
at w = 1.)

Guidance control — mean P(shot) assigned by an independent HGB shot
classifier trained on real corners:

| Corners scored                   | mean P(shot) |
|----------------------------------|--------------|
| real, no-shot corners            | 0.164        |
| generated, guided to shot=0      | 0.182        |
| generated, real labels           | 0.202        |
| generated, guided to shot=1      | 0.248        |
| real, shot corners               | 0.269        |

Interpretation:

1. **Outcome control works**: guiding to shot=1 vs shot=0 recovers ~78% of
   the real shot/no-shot separation as judged by an independent classifier —
   the diffusion analog of TacticAI's "adjust players to raise/lower shot
   probability" demo.
2. **Realism is close but not at paper level**: our discriminators can still
   partially tell (F1 ~0.62 vs their 0.53). `diffusion_distributions.csv`
   shows why: synthetic corners are slightly under-densified in the goalmouth
   (fewer players in the 6-yard box, larger min distance to goal, ~0.3 sd
   gaps) — classic diffusion over-smoothing of tight clusters.
3. Fairness caveats vs the paper: they trained on ~3x more corners of cleaner
   full-tracking data, and their MLP saw raw representations while our
   discriminator gets engineered global features (incl. velocity aggregates),
   a substantially stronger test. Scale-up (2x params, 2x steps) did not
   improve realism — the next lever is method, not size: v-prediction / EDM
   preconditioning, min-SNR loss weighting, self-conditioning, or flow
   matching, plus training the position-only model on all ~34k open-data
   corners (no 360 needed for positions of visible players... limited) or
   inpainting-style defender resampling.

Figure: `real_vs_synthetic.png` (3 held-out corners, real vs generated on the
same roster template).

### Extended quality metrics (`quality_metrics.py` -> `quality_metrics.csv`)

The discriminator test answers only "same distribution?". These five axes fail
independently, and each is reported against a **real-vs-real control** (the
holdout split in half), which is the honest yardstick: the control shows what
a *perfect* generator would score given finite samples.

Descriptor space = normalized histograms of att-att / def-def / att-def
pairwise distances (permutation-invariant, feature-engineering-free).

| metric | synthetic vs real | real vs real (control) | reading |
|---|---|---|---|
| MMD^2 (RBF, perm. test) | 0.0056, p = 0.001 | -0.0005, p = 0.62 | distributions differ significantly |
| - att-att / def-def / att-def | 0.0029 / 0.0066 / **0.0119** | - | the gap is mostly **attacker-defender** structure |
| 1-NN two-sample accuracy | 0.540 | 0.487 | only 4 pts above chance -- much closer than the feature-based discriminator suggested |
| precision (fidelity) | 0.829 | 0.811 | samples land on the real manifold as often as real ones |
| recall (diversity) | 0.810 | 0.868 | mild diversity loss |
| density | 1.057 | 0.840 | slight over-concentration in already-dense regions |
| coverage | 0.959 | 0.967 | real modes are covered -- no mode collapse |
| median Chamfer to nearest TRAIN corner | 2.04 m | 1.99 m (real holdout) | **ratio 1.02 -> no memorization** |

Physical plausibility (rates scored identically on real corners, so the base
rate from tracking noise is visible):

| check | real | synthetic |
|---|---|---|
| players beyond the goal line | 0.5% of players | 2.4% of players |
| players outside the side lines | 0.08% | 3.0% |
| corners with a sub-0.5 m overlap | 32% | 38% |
| corners with a >9 m/s player | 5.4% | 2.9% |

Overshoots are small (mean 0.30 m, max 2.0 m), so this is boundary *softness*,
not players flying off the pitch: diffusion blurs hard constraints the data
never violates. Cheap fix if it matters: clamp the x/y channels to the pitch
during sampling (the current clamp at +-2.5 in normalized units is far too
loose to act as a pitch boundary).

What this adds beyond the discriminator: the generator is **not copying**
(memorization ratio 1.02), **not mode-collapsed** (coverage 0.96), and its
fidelity matches the control -- but it measurably under-reproduces
**attacker-defender proximity**. The g(r) curves in `quality_metrics.png`
show it directly: at r < 2 m real corners have far more attacker-defender
pairs than synthetic ones (density 0.0062 vs 0.0043), while same-team pairs
are slightly *over*-clustered. In football terms the model reproduces where
players stand but softens **tight man-marking contact** -- consistent with the
earlier goalmouth-density finding, now localized to the specific interaction
that causes it.

Figure: `quality_metrics.png` (g(r) per pair type + nearest-training-corner
histogram). Raw curves: `quality_pcf.csv`.

### Positioning suggestions (paper Fig. 3 analog)

> **Superseded — read with the correction.** Every best-of-6 number in this
> section was produced by selecting the best candidate and measuring it with the
> *same* classifier, which inflates the gain by roughly +0.14. Cross-fitted
> (`tactic_recommend.py`), choosing among draws is worth nothing and the real
> effect of the guidance is about +0.03. The mechanism described here is sound;
> the effect sizes are not.


`tactic_suggest.py` / `tactic_suggest2.py` produce coaching-style adjustments
by **inpainting**: one team's players are pinned at their real positions while
the other team is re-denoised from noise, conditioned on the desired outcome.

- ATTACK suggestion : defenders + GK + taker frozen, outfield attackers
  re-generated with label **shot = 1** (guidance w = 2)
- DEFENSE suggestion: attackers + taker frozen, outfield defenders
  re-generated with label **shot = 0**

The taker stays fixed in both (it is the corner spot), and velocities are
regenerated together with positions, so a suggestion is "stand here, moving
this way". `tactic_suggest2.py` adds the paper's propose-and-evaluate loop:
the diffusion model proposes k = 6 candidate adjustments, the independent
shot predictor scores them, and the best (max P for attack, min P for
defense) is kept.

Mean P(shot) over the 485 holdout corners, scored by the shot classifier
trained on real training corners only:

| setup                          | mean P(shot) |
|--------------------------------|--------------|
| original (real corners)        | 0.185        |
| ATTACK suggestion, single draw | 0.220        |
| **ATTACK suggestion, best-of-6**   | **0.323**    |
| DEFENSE suggestion, single draw| 0.204        |
| **DEFENSE suggestion, best-of-6**  | **0.142**    |

Best-of-6 improves the target metric on **94%** of corners for attack and
**82%** for defense. On the 100 holdout corners that produced a real shot --
the paper's evaluation slice -- the defensive suggestions cut mean P(shot)
from **0.269 to 0.198** (paper: 0.75 -> 0.69 on their 100 threatening
corners; absolute levels are not comparable across datasets/predictors, the
direction and relative size are).

Note the single-draw defense result (0.204, slightly *above* the original):
a lone conditional sample is not reliably an improvement, because guidance at
w = 2 moves players tactically but also off-distribution. The paper's design
answer is exactly the loop implemented here -- generate several candidates,
let the predictor pick. That is why best-of-6 flips it to a solid reduction.

Figure: `tactic_suggestions.png` -- original / attack / defense for a
high-effect and a median corner, with gray circles and lines tracing each
player from original to suggested position. Per-corner numbers:
`tactic_suggestions.csv`; chosen player coordinates:
`tactic_best_players.parquet`.

**Selection caveat on every best-of-6 number above.** The candidate that gets
reported is chosen by the same classifier that then scores it, so the headline
is a maximum over 6 draws of a noisy score, not the typical effect of the
adjustment. Decomposing this on the joint recommender below (same architecture,
same k = 6, same scorer): the mean over candidates gains +0.025 while the max
gains +0.116, so **roughly 78% of a best-of-6 headline is selection** rather
than the guidance moving setups. The direction is real and the mean gain is
positive; the magnitude is not. The paper's design is also propose-and-evaluate,
so this is a property of the protocol, not a deviation from it -- but the
adjusted reading of "0.185 -> 0.323" is a genuine improvement of roughly a
quarter that size.

### Joint recommendation: shape + delivery (`tactic_suggest3.py`)

Answers a fuller coaching question than Fig. 3: given the defence as it stands,
**where should the attackers go AND what kind of ball should be delivered?**

**Technique is not a free variable.** Corner side crossed with the taker's foot
determines the swing, with no counterexample in 2,153 long corners:

| corner side | taker's foot | Inswinging | Outswinging | Straight | n |
|---|---|---|---|---|---|
| y=0 | Left | 0.0% | **93.3%** | 6.7% | 238 |
| y=0 | Right | **92.0%** | 0.0% | 8.0% | 765 |
| y=80 | Left | **94.0%** | 0.0% | 6.0% | 615 |
| y=80 | Right | 0.0% | **83.7%** | 16.3% | 535 |

The residual is `Straight`, a deliberately flat ball. So recommending a corner
type *is* recommending which foot takes it — an action a coach can take. After
the tacticai reflection this collapses to: effective right foot -> inswinging,
effective left foot -> outswinging.

(Note `foot_right` in `GDELIV` is the raw body part while the geometry is
mirrored, so unlike `tech_in`/`tech_out` it is not reflection-invariant — it
conflates two geometrically distinct cases. `tactic_suggest3.py` sets the
effective foot and derives the raw flag; the classifier configs still use the
raw one.)

**Is the delivery contrast trustworthy?** Outswingers show a 29.0% shot rate
against 18.8%, but teams pick the taker, so the gap is confounded. Treating the
delivery as the treatment and the freeze-frame setup as the adjustment set
(1,534 swung training corners, propensity AUC 0.841, 86% retained after
trimming to common support):

| contrast | outswinging − inswinging |
|---|---|
| raw difference in shot rate | +0.089 |
| IPW adjusted on the setup | **+0.062** |

30% of the raw gap was setup confounding. Personnel, game state and opponent
remain unadjusted, so 0.062 is an upper bound on a causal reading, not an
estimate of one.

**And the advantage does not survive a change of outcome.** Joining these
corners to the pipeline's `decision_points.parquet` on `uuid` (2,461 of 2,473
match) gives xG and goals for the same possessions. Cluster bootstrap over
matches, outswinging − inswinging:

| outcome | difference | 95% CI |
|---|---|---|
| shot-attempt rate | **+0.1025** | [+0.0618, +0.1431] |
| xG | +0.0019 | [−0.0091, +0.0128] |
| goal rate | +0.0011 | [−0.0197, +0.0235] |

The attempt advantage is large and solid; the value advantage is
indistinguishable from zero. The mechanism is visible per attempt —
inswingers produce 241 attempts worth 0.128 xG each converting at 12.9%,
outswingers 194 attempts worth 0.110 converting at 7.7% (small samples, so
treat the conversion split as the mechanism rather than a precise estimate).
Outswingers generate *more but worse* chances, and the two effects cancel. If
the attempt gain translated proportionally, xG would rise by roughly +0.012;
the observed +0.0019 sits well below that and the CI's upper bound barely
reaches it.

So the delivery recommendation below is chasing a quantity that is not worth
anything in goals. **`P(shot|corner)` is the right target for reproducing the
paper and the wrong objective for a tactical recommendation.**

**Results** (388 swung holdout corners, defence pinned, k = 6 shapes × 2 feet):

| stage | mean P(shot) | gain |
|---|---|---|
| original | 0.210 | — |
| + best attacking shape | 0.327 | +0.116 (94% of corners improve) |
| + best delivery | 0.360 | +0.033 (switches foot on 67%) |

Both halves need discounting, in different ways:

- **The shape gain is 78% selection.** Mean over the 6 candidates is +0.025
  against +0.116 for the max. The guidance does move setups in the right
  direction, but by about a quarter of what best-of-6 advertises.
- **The delivery half is a near-constant policy.** It recommends outswinging on
  **91%** of corners (241 of 257 observed inswingers get flipped). A policy that
  close to constant is reproducing the marginal shot-rate gap, not reading the
  defence in front of it — which is what you would expect, since the scorer sees
  technique as an 8-dimensional one-hot block with no interaction term against
  the defensive geometry. Conditioning is happening in the *shape* half; the
  delivery half is closer to "outswingers generate more shot attempts on
  average", already known from the marginal table.

Both gains are also measured in the scorer's units, not reality: guided samples
at w = 2 drift off-distribution, so a classifier trained on real corners is
being asked about setups unlike its training data.

The delivery half becomes genuinely conditional once the generator itself is
delivery-aware (see the ball-consistency section) — then technique enters the
setup rather than being pasted onto a fixed one, and "an inswinger to the near
post against THIS defence" is a query the model can answer instead of a column
in a one-hot.

Figure: `tactic_joint.png` — defence identical across both panels, attackers
moved, delivery switched.

### Where should the corner be delivered? (`delivery_target.py`)

Technique is near-determined and worth nothing in goals. The delivery *target*
is the decision that is genuinely free, so this asks it directly, scoring every
corner by xG and goals rather than shot attempts. 2,461 corners; frame mirrored
so the kick is at (120, 0), which fixes the goal, making near post always
y = 36 and far post y = 44. CIs are cluster bootstraps over matches.

**Lateral position is the lever.**

| lateral band | n | share | shot rate | xG | 95% CI | goal rate |
|---|---|---|---|---|---|---|
| **central** (between the posts) | 1,014 | 44.8% | 0.215 | **0.0561** | [0.0481, 0.0639] | 0.048 |
| far post | 299 | 13.2% | 0.415 | 0.0446 | [0.0367, 0.0533] | 0.040 |
| near post | 847 | 37.4% | 0.176 | **0.0374** | [0.0311, 0.0443] | 0.034 |
| beyond far post | 104 | 4.6% | 0.067 | 0.0296 | [0.0111, 0.0550] | 0.029 |

Central is worth ~50% more xG than the near post and the intervals do not
overlap. It holds within depth bands too: at 0-4 yd out, central is 0.0574
[0.0407, 0.0747] against 0.0304 [0.0217, 0.0412] for the near post.

**Depth barely matters.** 9 yd+ 0.0484, 4-9 yd 0.0475, 0-4 yd 0.0429 — heavily
overlapping. But shot-attempt rate collapses with depth (0.315 / 0.260 /
0.079), because deliveries landing on top of the goal get claimed by the
keeper. Another case of attempts and value coming apart: judged on attempts you
would never deliver deep, judged on xG it costs little.

**The conditional effect is weak.** Splitting by defenders in the six-yard box
at the kick, central is still the best band whether the box is packed (0.0534,
n=632) or open (0.0604, n=382). A packed box narrows the central advantage and
slightly favours the near post (0.0389 vs 0.0351) and beyond the far post
(0.0390 vs 0.0158, n=62/42 so noisy), but nothing overturns central.

**Technique interacts with attempts, not value.** Delivered centrally,
outswingers produce a 0.346 shot-attempt rate against 0.149 for inswingers —
and 0.0600 xG against 0.0548. The attempt gap is enormous, the value gap is
nothing, exactly as in the marginal contrast.

**The recommendation**: deliver central, between the posts, and worry less
about depth than about lateral placement. 37% of long corners currently go to
the near post — the lowest-value band of the three main ones.

**How much to trust it.** `pass.end_location` is where the ball *ended*, not
where it was aimed, so a zone's value bundles aim, execution, and whether the
defence got there first. "Near post is worse" partly means "the near post is
where defenders are". That is still decision-relevant — aiming into the crowd
is genuinely bad — but it is not `do(aim at zone Z)`, and no zone ranking here
is adjusted for defensive setup beyond the explicit split above. Establishing
the causal version needs the corner analog of `estimand_box.py`: target zone as
treatment, freeze-frame geometry as the adjustment set, positivity checked per
zone.

Figure: `delivery_target.png` — xG and shot-attempt surfaces over the arrival
point, side by side, which is the clearest picture of how far the two come
apart.

### Corner recommendation with the ball path (`tactic_recommend.py`)

`tactic_suggest3.py` maximised P(shot) and concluded "switch the taker's foot on
67% of corners, outswing on 91%". That target was faithful to the paper's Task 2
but wrong as advice. Adjusted for the defensive setup and bootstrapped over
matches, outswinging buys:

| outswinging − inswinging | IPW-adjusted | 95% CI |
|---|---|---|
| shot rate | **+0.105** | [+0.052, +0.154] |
| xG | +0.003 | [−0.009, +0.016] |
| goals | −0.003 | [−0.024, +0.021] |

Real on attempts, absent on value. Inswinger attempts are worth ~0.25 xG each
against ~0.17 for outswingers — the extra outswinger shots are cheaper shots.
`tactic_recommend.py` therefore recommends on xG and on the zone table, and
reconstructs the ball path for both the observed and the recommended delivery
via the pipeline's `ball_trajectory.reconstruct`, so a recommendation carries a
flight time and an apex rather than only an end point.

Recommended target, from 584 real deliveries that landed there: **(113.5, 39.6)
— central, 4–9 yd off the goal line**, median flight 1.68 s, reconstructed apex
4.64 yd. Only 23% of holdout corners were already delivered into that zone.

The recommendation **keeps the taker**, so the technique mix of the output is
just the observed one — 60.5% inswinging, 30.8% outswinging, 8.7% straight. It
is not "always inswing"; it is "no xG case for switching".

#### Why the target is constant, not per-corner

The recommended point is the same for every corner, which is the same shape of
output this project criticised in `tactic_suggest3.py`. It is kept constant on
purpose: a per-corner target is not supported by the data.

Fitting `(defensive setup x target band) -> outcome` on the training corners and
testing on holdout:

| target | out-of-sample |
|---|---|
| shot | AUC 0.557 (0.5 = nothing) |
| xG | **R² −0.074** (constant-mean baseline −0.002) |
| goal | AUC 0.581 |

The xG model is *worse than predicting the training mean*. Such a model still
happily emits per-corner argmaxes — 264 central, 136 far post, 43 near post on
holdout — but the median predicted gap between its best band and central is
**0.0000**, and it has no out-of-sample signal to produce that variation from.
It would be noise dressed as personalisation.

The reason is visible in the outcome itself: 62% of corners produce zero xG and
the standard deviation is 2.7x the mean. Band-level differences of 0.01-0.02 xG
are real in aggregate — the bootstrapped CIs in `delivery_target_zones.csv`
support them — but they are an order of magnitude below the per-corner noise.
The honest resolution of this data is a zone-level policy, not a per-corner one.

`tactic_recommendation_gallery.png` shows the recommendation across the six most
common delivery zones. The target is identical in all six; what varies is how
far the ball has to travel to reach it (1.3 yd when the corner was already
central, 10.8 yd from a deep far-post delivery) and how the attacking shape
responds.

#### Why keep the technique rather than pick one (`technique_effect.py`)

"Keep" is a decision under a null, so it needs defending on two fronts: that the
null is real, and that keeping is the right response to it.

**Is the null hiding an interaction?** If the swing mattered in *some* defensive
state, a conditional recommendation would beat a constant one. Scanned across
nine strata — six defensive splits and three target bands, including the
mechanistically plausible ones (zonal marking, packed six-yard box):

| | result |
|---|---|
| strata with an xG effect | **0 of 9** (≈0.5 expected by chance) |
| strata where outswinging raises the shot rate | **8 of 9** |

The swing buys attempts almost everywhere and value nowhere. There is no state
in which switching is indicated.

**Why keep, given that?** Because switching swing is not a free relabeling —
side × foot fixes the swing, so it means switching *takers*, and the designated
taker is usually the best striker of the ball. The scorer sees geometry and
cannot see delivery quality, so it systematically understates that cost. Under
no measured benefit, the action carrying unmodelled downside loses. The observed
60/31/9 split also encodes who can actually hit a corner, which the freeze frame
does not observe.

Stated actively, the recommendation is: **choose the taker on delivery quality,
not on swing type, and spend the tactical decision on the target zone instead**
— which does move xG.

**Two honest limits.** Null is not equivalence: the CI [−0.010, +0.016] on a
base of 0.047 still admits an effect of about a third of the base rate in either
direction. And "keep" is conditional on caring about xG. The shot-rate effect is
robust (+0.105, positive in 8 of 9 strata), so a team that specifically wants
more attempts — chasing a game, hunting rebounds and second balls — has a real
reason to outswing. It just does not convert into goals in this data.

#### Best-of-N adds nothing: the selection bias, measured

Picking the best of N candidates with the same model that then reports the gain
is selection on that model's own noise — the argmax of N noisy scores is biased
upward even when every candidate is identical in truth. Cross-fitting removes
it: train a **selector** and an **evaluator** on disjoint halves of the training
corners, let the selector choose and the evaluator measure. Over 444 long
holdout corners:

| how the shape gain is measured | mean gain |
|---|---|
| one arbitrary draw, no selection | +0.034 |
| mean over all 6 draws | +0.028 |
| **best-of-6, measured independently** | **+0.026** |
| best-of-6, measured by the selecting model | +0.170 |

**Choosing among draws is worth −0.003 — nothing.** The independently measured
best-of-6 is no better than grabbing one draw at random, and the entire apparent
advantage of best-of-6 (+0.144 of the +0.170) was the selecting model marking
its own homework. What survives is the guidance itself: a single conditional
draw with the defence pinned is worth about **+0.03** shot probability, roughly
a fifth of what best-of-6 appeared to deliver.

This applies directly to the earlier `tactic_suggest.py` numbers below
(0.185 → 0.323 best-of-6, "improved on 94% of corners"): those are single-model
figures and carry the same inflation. Only cross-fitted numbers should be
quoted.

### Full-set delivery analysis (`delivery_target_full.py`) — REVISES the above

Everything above ran on the 2,473 corners in 360-covered matches, because that
is where the freeze frames are. The zone question does not need freeze frames —
only where the delivery arrived and what the possession was worth — so it can
run on all 4,235 matches.

Corners are identified in `decisions.parquet` as the first decision of a
`From Corner` possession taken from the corner arc. Validated against the
exactly-identified subset: **2,472 of 2,473 recovered (100.0%)**. That gives
**41,178 corners, 16.6x the subset**, and CIs **4.3x tighter** (median 95% width
per zone 0.0309 → 0.0071).

**The band ranking does not survive.**

| target | n | xG (full set, 95% CI) | xG (360 subset) |
|---|---|---|---|
| far post | 5,283 | **0.0480** [0.0453, 0.0509] | 0.0446 |
| central | 15,220 | **0.0458** [0.0442, 0.0473] | 0.0561 |
| near post | 14,414 | 0.0357 [0.0341, 0.0373] | 0.0374 |
| beyond far post | 1,788 | 0.0160 [0.0129, 0.0197] | 0.0296 |

Paired, cluster-bootstrapped over matches:

| contrast | full set |
|---|---|
| central − far post | −0.0022 [−0.0054, +0.0009] — **crosses 0** |
| central − near post | +0.0101 [+0.0078, +0.0123] * |
| central − beyond far post | +0.0298 [+0.0261, +0.0331] * |

Central and far post are **statistically indistinguishable**, with far post
nominally ahead. The subset's apparent central dominance was small-sample noise
— and the recommender was built on it.

What hardened instead:

- **The near post is the finding.** Worse than central by +0.0101 with a tight
  CI, on n=14,414 — and it still takes **39.3% of all long corners**. This is
  the same conclusion as the subset, now on 17x the data.
- **Depth matters more than the subset suggested.** The best zones are the 4–9 yd
  band (`4-9yd x far post` 0.0509, `4-9yd x central` 0.0496); delivering to
  0–4 yd — right on top of the goal, in the keeper's reach — collapses to 0.0351
  central and 0.0279 near post. On the subset `0-4yd x central` looked like
  0.0574. That reversal is the clearest small-sample casualty.

`tactic_recommend.py`'s target was widened from central-only `(36, 44)` to
`(36, 52)` — the 4–9 yd band across central through far post — since picking a
point inside one of two indistinguishable zones is false precision.

**Technique on the full set** (28,805 long swung corners, 12,282 out /
16,523 in) confirms the earlier null and tightens it a long way:

| outswinging − inswinging | full set | 360 subset |
|---|---|---|
| xG | +0.0025 [−0.0000, +0.0049] | +0.0031 [−0.0098, +0.0158] |
| goals | +0.0006 [−0.0036, +0.0051] | −0.0028 [−0.0236, +0.0204] |

Still crossing zero, but the bound is now ~12% of the base rate rather than the
~35% the subset could manage. "Pick the taker on delivery quality, not swing" is
much better supported than it was.

**What the full set cannot do:** no freeze frames outside the 360 subset, so
nothing here conditions on the defensive setup, and the technique contrast is
unadjusted (the IPW-adjusted version on the subset agreed, which is the check
available). Defence-conditional results stay on the smaller sample.

#### Why each zone scores: retention × payoff

`pass.end_location` is where the ball *arrived*, and the defence has a say in
that, so a zone's value is the product of how often the delivery survives to an
attacker and what it is worth once it does (`delivery_risk_reward.csv`):

| target | n | reaches an attacker | xG (uncond.) | xG if won | shot if won |
|---|---|---|---|---|---|
| central | 1,014 | **27%** | **0.0561** | **0.1347** | 0.84 |
| far post | 299 | **60%** | 0.0446 | 0.0668 | 0.71 |
| near post | 847 | 42% | 0.0374 | 0.0668 | 0.44 |
| beyond far post | 104 | 87% | 0.0296 | 0.0474 | 0.11 |

Three things fall out:

- **Central is high risk, high reward.** It is the hardest zone to win — the
  defence gets it 73% of the time, unsurprisingly, since that is where they and
  the keeper are — and double the payoff of anywhere else when it lands. Its
  top unconditional value is that product, not a high success rate.
- **The near post is dominated.** Far post reaches an attacker 60% of the time
  against the near post's 42%, for an identical payoff once it does (0.0668
  both). There is no defensive state in which the near post is the answer, and
  it still absorbs ~32% of all corners.
- **Beyond the far post is the safe nothing.** 87% retention, a 11% shot rate.
  Useful to keep the ball, useless to score.

The conditional column selects on an outcome (who touched it first), so it is a
mechanism diagnostic; the unconditional column is the expected value to act on.

#### Where the xG numbers come from

Nothing here fits an xG model. The per-shot value is **StatsBomb's own**
`shot.statsbomb_xg`, taken off the raw event. The pipeline
(`stage234_possessions.py`, `extract_decisions`) then sums it forward:

    xg_after(decision i) = sum of xg over decisions i .. end of possession

restricted to events by the possession team, with the possession boundary being
StatsBomb's own `possession` id. A corner is a Pass and so contributes `xg = 0`
itself; its `xg_after` is entirely the shots that follow. A zone's number is the
mean `xg_after` over corners whose `pass.end_location` fell in that zone.

**So the zone value is the whole possession, not just the header.** After a
corner the possession runs a median of 2 further decisions (mean 5.1, max 119),
and 24.6% of corners are the last decision of their possession. A cleared corner
that is recycled and shot still counts, as long as StatsBomb did not switch the
possession id.

Decomposing it: the first shot after the corner is worth 0.0343 of the 0.0448
mean total, so **77% of the credited value is the first attempt** and 23% is
second phase. Corners average 0.47 shots per possession and only 5.3% produce
more than one.

The band ranking is unchanged on first-shot-only xG, so the recommendation is
not an artifact of crediting long possessions to the delivery:

| lateral band | xG (full possession) | xG (first shot only) | shots/possession |
|---|---|---|---|
| central | 0.0558 | **0.0416** | 0.49 |
| far post | 0.0446 | 0.0361 | 0.65 |
| near post | 0.0373 | **0.0274** | 0.43 |
| beyond far post | 0.0296 | 0.0280 | 0.22 |

If anything the central-over-near-post ratio is slightly wider on first shots
(1.52x vs 1.50x). Note the far post again generates the most attempts per
possession while ranking below central on value — the same attempts/value split
that runs through the technique and depth results.

Two caveats on the metric itself. StatsBomb's xG is a model, and corners are
among its harder regimes: crowded six-yard boxes and headers under contact are
where any xG model carries the most error. And crediting a delivery zone with
everything downstream in the possession is generous by construction — the
first-shot column above is the conservative reading.

#### Quantifying the defence (`defense_features.py`)

The split above describes the defence with one number — visible outfield
defenders in the six-yard box — which cannot tell a flat line from three men on
the posts. `defense_features.py` builds a fuller description from the same
freeze frame and re-tests the conclusion.

| descriptor | what it captures | r with `def_in6` |
|---|---|---|
| `mark_dist` | mean distance defender → nearest attacker: low = man-marking, high = zonal | −0.224 |
| `marked_frac` | share of defenders within 2 yd of an attacker | 0.128 |
| `def_spread_y` | lateral dispersion — compact block vs spread | −0.199 |
| `def_line_x` | how deep the defensive line sits | 0.781 |
| `near_post_cover` / `far_post_cover` | defenders within 3 yd of each post | 0.497 / 0.362 |
| `gk_depth` | keeper's distance off the goal line | −0.418 |
| `def_in6_frac` | in-box count as a share of visible defenders | 0.963 |

Marking scheme is not recorded anywhere in StatsBomb, but it is inferable from
geometry, since man-marking puts a defender next to a body and zonal puts one
in a space. It turns out to be a **continuum, not a dichotomy**: `mark_dist`
runs 1.96 yd (p5) to 5.55 yd (p95) with no bimodality, and 44% of defenders sit
within 2 yd of an attacker on average — hybrid systems, as in the modern game.
So "zonal vs man" cannot be treated as a binary label here.

**Value does move with the defence**, mildly: more man-marking concedes more
(xG 0.0488 in the low `mark_dist` tercile vs 0.0423 in the high, and
`marked_frac` agrees in direction), and a keeper who stays on the line concedes
less (0.0387 vs 0.0471–0.0527).

**But the central recommendation is robust.** Splitting the delivery-target
table on `mark_dist`, `def_in6`, `near_post_cover` and `def_line_x` in turn,
central is the highest-xG band in all eight halves. The earlier "no conditional
effect" was not an artifact of the crude measure.

The one conditional signal that recurs across two independent measures: against
a packed six-yard box, deliveries **beyond** the far post improve sharply
(0.0507 vs 0.0170), consistent with the 0.0390 vs 0.0158 split in the section
above. Both rest on n < 65, so it is a hypothesis, not a finding.

**Visibility**: 360 frames only hold players in camera view, but at corners the
camera is on the box and defender coverage is near complete — median 10 of 10
outfield defenders, `corr(visible, in-box) = 0.234`. Packed corners average 9.67
visible defenders against 9.27 for open ones, so framing accounts for roughly a
tenth of the gap between the groups rather than creating it. Counts are also
reported as fractions of visible defenders (`def_in6_frac`) so the measures do
not silently track camera coverage. The principled fix, unused here, is the
pipeline's `visible_area.parquet` — the camera polygon per event — which
separates "no defender there" from "cannot see there".

Not captured: defender velocities (available and unused — these are static
snapshots at the kick), block structure beyond dispersion, and anything about
personnel. None of these descriptors are adjusted for confounding; they
describe, they do not identify.

### TSTR: train on synthetic, test on real (`tstr_eval.py` -> `tstr_results.csv`)

One synthetic corner generated per real training template (w = 1, labels =
the conditioning labels), then the Task-2 direct shot classifier is trained
on each source and evaluated on the same 485 REAL holdout corners
(mean +- std over 3 classifier seeds; thresholds tuned on each source's own
validation slice, so TSTR never sees real data before test):

| training source              | n     | AUC             | F1              |
|------------------------------|-------|-----------------|-----------------|
| TRTR: real corners           | 1,940 | 0.719 +- 0.010  | 0.447 +- 0.010  |
| TSTR: synthetic corners only | 1,940 | 0.693 +- 0.008  | 0.408 +- 0.007  |
| AUG: real + synthetic        | 3,880 | 0.696 +- 0.007  | 0.419 +- 0.008  |

Reading:

- A classifier that has NEVER seen a real corner keeps ~88% of the real
  model's above-chance discriminative signal (0.693 vs 0.719 AUC, chance
  0.5). The generator captures most of the decision-relevant structure --
  which players where, moving how, makes shots likely -- despite the
  detectable distribution shift the discriminators exploit.
- Augmentation does not help (0.696 vs 0.719): the synthetic set is a
  re-expression of the same 1,940 templates, so it adds smoothing noise, not
  new information. To help downstream training, synthetic data would need
  novel templates (sampled rosters) or a generator trained on more data.

### Batch-size ablation (`ablate_batch.py` -> `ablation_batch.csv`)

Same architecture, split, and 30k-step budget; only the training batch size
varies. Realism = discriminator on held-out real vs synthetic (lower = more
realistic, 0.5 = chance); guidance = mean P(shot) under shot=0 / shot=1
guidance from the independent classifier.

| batch | MLP F1 | MLP AUC | HGB F1 | HGB AUC | P shot0 -> shot1 |
|-------|--------|---------|--------|---------|------------------|
| 64    | 0.65   | 0.71    | 0.65   | 0.74    | 0.18 -> 0.25     |
| 128   | 0.60   | 0.68    | 0.62   | 0.69    | 0.18 -> 0.25     |
| 256   | 0.65   | 0.69    | 0.66   | 0.73    | 0.18 -> 0.25     |
| 512   | 0.57   | 0.62    | 0.66   | 0.72    | 0.18 -> 0.25     |

Reading (single seed per cell, so differences under ~0.04 are within the
noise band observed across seeds elsewhere in this project):

- Guidance separation is completely flat across batch sizes — outcome
  controllability does not depend on this hyperparameter.
- No consistent realism winner: the MLP ranks batch 512 best, the stronger
  HGB ranks batch 128 best, and batch 64 is worst for both. Batch size is
  not the lever that closes the realism gap (consistent with the earlier
  finding that 2x model / 2x steps did not help either); the residual
  goalmouth under-densification survives every setting tested.
- Checkpoints for all four runs are kept as `models/diffusion_model_b{64,128,256,512}.pt`
  (loadable via `load_model(tag="_b128")` etc.).

### Pitch-projection ablation (`ablate_projection.py` -> `ablation_projection.csv`)

`quality_metrics.csv` reports out-of-bounds violations in **75.9%** of
synthetic corners against **8.9%** of real ones, which reads like a large
constraint failure and a candidate explanation for the realism gap. It is not.
That metric is a per-*set* "any player" indicator, and a corner has ~20
players, so it amplifies a modest per-player difference. Per player the rates
are 4.18% synthetic vs 0.55% real, and the *magnitudes* nearly coincide
(median excess past the line 0.28 yd synthetic vs 0.32 yd real, p95 1.08 vs
1.25). The model puts players off the pitch about 7x too often, but never by
much more than the real data does.

`normalize()` maps the pitch onto exactly `[-1,1]^2`, so the sampler's
`x0.clamp(-2.5, 2.5)` was 2.5x the pitch in every direction — a numerical
guard, not a physical constraint. `Diffusion.project_x0(project=True,
margin_yd=...)` adds a real one, applied at every DDIM step (x0 feeds back
into x_{t-1}, so the constraint steers the trajectory rather than truncating
the answer). Same checkpoint, same sampling seeds, 10 discriminator splits
per arm; only the projection differs.

| arm | oob players | oob corners | MLP F1 | MLP AUC | HGB F1 | HGB AUC | 6-yd box |
|---|---|---|---|---|---|---|---|
| off (baseline) | 4.18% | 75.9% | 0.626 | 0.684 | 0.663 | 0.716 | 28.7% |
| margin 3.25 yd | 4.00% | 72.6% | 0.637 | 0.687 | 0.660 | 0.722 | 28.6% |
| strict pitch | 0.00% | 0.0% | 0.650 | 0.704 | 0.678 | 0.737 | 27.8% |
| REAL holdout | 0.55% | 8.9% | — | — | — | — | **35.1%** |

Reading:

- **The constraint is not the realism lever.** A margin of 3.25 yd (the real
  data's own out-of-bounds extent) almost never binds, and the strict pitch
  removes 100% of violations while moving every discriminator metric the
  *wrong* way — +0.025 MLP F1, +0.020 MLP AUC, +0.015 HGB F1, +0.021 HGB AUC.
  Each is inside 2 sd of the paired per-seed difference taken alone, but all
  four agree in sign, so the honest summary is "no better, plausibly slightly
  worse", not "no effect".
- The likely reason strict projection *hurts*: real freeze frames do contain
  off-pitch players, so generating exactly zero is its own distributional tell,
  and clamping piles mass onto the boundary line as a spike.
- Guidance separation is untouched (gap 0.066 / 0.069 / 0.062), so the
  constraint costs no controllability — it simply buys nothing.
- **The goalmouth gap is the real defect and no projection variant touches
  it.** Synthetic corners put 28.7% of players in the 6-yard box against 35.1%
  real — a 6.4 pp shortfall — while 18-yard box occupancy is already correct
  (82.8% vs 81.4%). The model gets the right number of players into the box
  and then fails to concentrate them at the near post. That is ~10x the
  boundary effect in magnitude and it is where the remaining realism gap lives.
- Sampling defaults are unchanged (`project=False`), so every result above
  this section still reproduces exactly; the baseline arm here is bit-identical
  to `diffusion_eval.py`'s numbers.

Since capacity, training steps, batch size, and now hard constraints have all
failed to close it, the next hypothesis worth testing is the **coordinate
parameterization**: absolute (x, y) under an epsilon-MSE objective smooths
exactly the sharp, multimodal density that the 6-yard box consists of. A
goal-relative polar encoding (distance and angle to goal) would spend
resolution where the density is sharpest.

### Ball trajectories on synthetic corners (`synthetic_trajectories.py`)

The generator emits player positions and velocities, not a delivery. Every
synthetic corner is sampled on a real corner's roster template, so the ball
that belongs to that template *is* the ball for the generated setup: the model
is answering "where would the players stand for THIS corner", with the corner
held fixed. Reconstruction physics is imported from the pipeline's
`ball_trajectory.py` rather than reimplemented, so its assumptions (swing
magnitude is a prior, contact height is a per-category prior) carry over
unchanged. Frames reconcile cleanly: `corner_id` is the StatsBomb event uuid,
`corners.parquet` carries the `reflected` flag, and the y-reflection leaves the
goal centre (120, 40) fixed, so the swing geometry is invariant under it.

That turns the ball into a test the project did not have. A real corner is a
configuration of players organised *around* a delivery. If the generator has
learned that structure, its players should converge on the arrival point the
way real ones do.

Distance from the ball's arrival point to the nearest attacker, 485 holdout
corners, same delivery for both rows:

| setup | median at kick | within 3 yd at kick | median at arrival | within 3 yd at arrival |
|---|---|---|---|---|
| real | 3.14 yd | 46.8% | 1.36 yd | **73.0%** |
| synthetic (real label) | 4.32 yd | 32.8% | 4.17 yd | **33.8%** |
| synthetic (shot=1) | 4.80 yd | 29.3% | 4.33 yd | 30.3% |
| synthetic (shot=0) | 4.15 yd | 34.0% | 4.11 yd | 33.6% |

"at kick" uses freeze-frame positions; "at arrival" advances every player by
`v * flight_time`, which is the matched interval — velocities are ball-flight
averages, so `x + v*T` lands a player roughly where they were when the ball
got there.

**Caveat, stated up front**: real velocities are estimated from where players
actually ended up, so the real *arrival* figure is partly definitional. That is
why the kick-time column matters — it involves no velocity at all, and it shows
the same gap (46.8% vs 32.8%). The two measures agree, so the finding survives
the circularity.

Reading:

- Real setups put an attacker within 3 yd of the arrival point on 73% of
  corners. Generated setups, on the *same* deliveries, manage 34% — less than
  half. The median nearest attacker is 2.8 yd further away.
- This is a far larger defect than anything the realism discriminators
  register. They run at 0.68-0.72 AUC on aggregate features and call the
  distributions close, because a delivery-agnostic crowd has the right overall
  shape. Aggregate features cannot see that the crowd is in the wrong place
  *for this particular ball*.
- **The cause is a missing conditioning variable, not a capacity shortfall.**
  `Denoiser.forward` takes per-slot flags (teammate / keeper / actor / has_vel)
  and the shot label. It is never told the delivery — not the end location, not
  the technique, not the height. The model cannot place players around a ball
  whose destination it has no access to, so it generates the setup averaged
  over all deliveries the roster could receive.
- That plausibly explains **the goalmouth gap too**. Averaging over destinations
  smears mass that a specific delivery would concentrate, which is exactly the
  6-yd-box shortfall (28.7% vs 35.1%) that capacity, steps, batch size and hard
  constraints all failed to move. One missing input accounts for both symptoms.

The fix this points to is conditioning the denoiser on the delivery — end
location (or target zone), technique and height, alongside the existing shot
label. It is a retrain, but a better-motivated one than the coordinate
reparameterization: it addresses a variable the model demonstrably lacks rather
than a representation it might handle better. It also makes the generator far
more useful, since "show me how the defence should set up against an inswinger
to the near post" becomes a query it can answer.

Figure: `synthetic_trajectories.png` — three high deliveries, real above,
generated below, ball path in orange with dot size tracking height and X at the
arrival point.

## Files

Every script, grouped by role, with its status. **Superseded scripts are kept
deliberately** — every result in this README, including the ones later
overturned, is reproducible from the script that produced it. Status meanings:

- **current** — the live version for its job
- **current, library** — imported by other scripts, not run directly
- **superseded** — an earlier iteration; its results are quoted here only where
  marked as such
- **superseded, live import** — the results are stale but other current scripts
  import code from it, so it must not be deleted

### 1. Corner extraction — 360 subset (has freeze frames)

| File | Status | What it is |
|---|---|---|
| `paths.py` | current, library | Resolves the StatsBomb checkout (`$OPEN_DATA` → repo root → Desktop → Downloads). Import this rather than hardcoding a data path. |
| `outdirs.py` | current, library | Creates `results/` `figures/` `models/` on import. Deliberately separate from `paths.py`, whose import resolves the StatsBomb checkout and exits if it is missing — the torch-only scripts must not inherit that dependency just to get a directory. |
| `check_paths.py` | current, check | Static guard on the layout: walks every script's AST and flags a write that does not land under `results/` `figures/` `models/`. `--strict` exits 1. Catches the silent failure — an unmigrated write still succeeds, it just leaves the committed copy stale. |
| `extract_corners.py` | current | Builds the Task-2 corner dataset from raw events: labels, receiver matching, freeze-frame normalisation, velocity estimation from consecutive frames. → `corners.parquet`, `corner_players.parquet` |
| `onpitch.py` | current | Reconstructs the on-pitch XI and keeper at every corner from Starting XI / Substitution / Tactical Shift / red cards. → `onpitch.parquet` |
| `fetch_bio.py` | current | Wikidata height (P2048) and weight (P2067) for all 4,079 players. ~30 min, hits the API. → `player_bio.parquet` |
| `add_delivery.py` | current | Delivery attributes (technique / height / body part / length) for the 2,473 subset corners. → `corner_delivery.parquet` |

### 2. Corner extraction — full dataset (no freeze frames)

Not newer versions of the above — a **different scope**. These cover all 4,235
matches and 41,178 corners, at the cost of having no 360 data, so nothing built
on them can condition on the defensive setup.

| File | Status | What it is |
|---|---|---|
| `extract_corners_full.py` | current | Identifies every corner in the full dataset from `decisions.parquet` (first decision of a From Corner possession, taken from the corner arc). Validated: recovers 2,472 of the 2,473 exactly-identified subset corners. → `corners_full_raw.parquet` |
| `extract_delivery_full.py` | current | Delivery attributes for all 41,147 corners, parsed from 12 GB of raw JSON in a process pool. → `corner_delivery_full.parquet` |

### 3. Task 2 models (shot prediction from corners)

| File | Status | What it is |
|---|---|---|
| `train_eval.py` | current | Features, receiver model, both shot models, Eq. 1, 5-seed evaluation of all 9 configurations. Defines `CONFIGS` and `build_features`, which most later scripts import — note `CONFIGS` values are 4-tuples since the `g` fix; downstream scripts read `[1]` (the shot model's globals) and are unaffected. `add_globals` broadcasts globals into the receiver model; `--check-cancel` proves they can only act through the tanh layer. → `results/results_seeds.csv`, `results/results_summary.csv` |

### 4. Diffusion generator

| File | Status | What it is |
|---|---|---|
| `diffusion_corners.py` | current | The set-diffusion model: data prep, permutation-equivariant denoiser, training with the laptop time-budget gate, DDIM sampler with optional pitch projection. Also `pick_device()` (CUDA → MPS → CPU) — don't reintroduce a hardcoded device. → `models/diffusion_model.pt`, `results/diffusion_split.csv`, `results/diffusion_loss.csv` |
| `diffusion_eval.py` | current | Sampling, discriminator realism test, guidance check, distribution table, figure. → `synthetic_corners.parquet`, `results/diffusion_eval.csv`, `results/diffusion_distributions.csv`, `figures/real_vs_synthetic.png` |
| `quality_metrics.py` | current | Extended generative metrics: MMD², 1-NN, precision/recall/density/coverage, constraint violations. → `results/quality_metrics.csv`, `results/quality_pcf.csv`, `figures/quality_metrics.png` |
| `tstr_eval.py` | current | Train-on-synthetic, test-on-real for the shot classifier. → `results/tstr_results.csv` |
| `gallery.py` | current | Real vs shot=0/shot=1-guided generations per template, plus a sample-diversity row. → `figures/synthetic_gallery.png` |
| `synthetic_trajectories.py` | current | Reconstructs ball paths for synthetic corners (importing the pipeline's `ball_trajectory`) and measures ball-consistency: does a generated setup converge on where the ball actually arrives. → `synthetic_corner_trajectories.parquet`, `results/ball_consistency.csv`, `figures/synthetic_trajectories.png` |

### 5. Ablations (both negative results)

| File | Status | What it is |
|---|---|---|
| `ablate_batch.py` | current | Batch-size ablation, 4 arms at a fixed 30k-step budget. → `results/ablation_batch.csv`, `models/diffusion_model_b{64,128,256,512}.pt` |
| `finish_ablation.py` | helper | One-off: evaluates the batch-512 checkpoint and merges it into `results/ablation_batch.csv`. Not an entry point — it runs at import. |
| `ablate_projection.py` | current | Pitch-projection A/B: one checkpoint, one set of sampling seeds, three constraint arms, 10 discriminator splits each. Shows the constraint buys no realism. → `results/ablation_projection.csv` |

### 6. Recommendation — read this lineage carefully

Four iterations of the same idea. **Only the last is live.**

| File | Status | What it is |
|---|---|---|
| `tactic_suggest.py` | **superseded, live import** | v1. Single guided draw, defenders pinned. Its best-of-6 numbers carry the selection bias, but `tactic_suggest2/3` and `tactic_recommend` all import `ddim_inpaint` and `draw_panel` from it. **Do not delete.** → `tactic_suggestions.csv/.png` |
| `tactic_suggest2.py` | superseded | v2. Best-of-k candidate selection — the mechanism that turned out to be the bias. Overwrites `tactic_suggestions.csv/.png`, so whichever of v1/v2 ran last owns those files. → `tactic_suggestions.csv/.png`, `tactic_best_players.parquet` |
| `tactic_suggest3.py` | superseded | v3. Joint shape + delivery, conditioned on the defence. Wrong on two counts: it maximised P(shot) (which moves attempts but not xG) and it selected and scored with one model. → `results/tactic_joint.csv`, `figures/tactic_joint.png` |
| **`tactic_recommend.py`** | **CURRENT** | The live recommender. xG-based rather than P(shot); cross-fitted selector/evaluator so the reported gain is unbiased; reconstructed ball path attached to every recommendation. → `results/tactic_recommendation.csv`, `figures/tactic_recommendation.png`, `figures/tactic_recommendation_gallery.png` |

### 7. Delivery and defence analysis

| File | Status | What it is |
|---|---|---|
| **`delivery_target_full.py`** | **current** | Authoritative zone analysis on all 41,178 corners. Supersedes the zone ranking in `delivery_target.py`. → `results/delivery_target_full.csv`, `figures/delivery_target_full.png` |
| `delivery_target.py` | **partly current** | Zone ranking superseded by the full set — but its defence-conditional splits and the retention × payoff decomposition need freeze frames and exist nowhere else. → `results/delivery_target_zones.csv`, `results/delivery_risk_reward.csv`, `figures/delivery_target.png` |
| `defense_features.py` | current | Defensive descriptors from the freeze frame: six-yard congestion, defensive line, post coverage, keeper depth, and a zonal-vs-man marking proxy (mean defender→nearest-attacker distance). → `defense_features.parquet/.csv` |
| `technique_effect.py` | current | Inswinging vs outswinging on attempts / xG / goals: raw, IPW-adjusted, cluster-bootstrapped, plus a nine-stratum interaction scan. The evidence the delivery recommendation rests on. → `results/technique_effect.csv`, `results/technique_interactions.csv` |

### Things that can mislead you

A full audit of every file in the project for names, keys and types that look
interchangeable but are not, plus one bug that silently changed published
numbers. Seven categories, in rough order of how much damage they can do —
`#0` first because it is the only one that has already caused a wrong number
to be written down.

#### 0. Task 2 numbers published before 2026-08-28 do not reproduce — fixed

`train_eval.py` split corners into `set`s of string ids and then *iterated*
them to build the feature matrices. Set iteration order over Python strings
depends on `PYTHONHASHSEED`, which is randomised per process, so row order
changed between runs; `HistGradientBoostingClassifier(early_stopping=True)`
holds out its validation rows **by position**, so the same seed trained a
different model each time. `auc_direct` on `vel` seed 0 ranged over
0.714–0.741 across four hash seeds — wider than the gaps between the
configurations being compared — while `recv_top3`, which does not depend on
row order, was stable to four decimals. That stability is what made it
invisible: the receiver metrics agreed run to run, so the shot metrics looked
trustworthy too.

Fixed by keeping the three collections as ordered arrays (they were only ever
iterated; `valid_recv` is still a set, since it is used only for membership).
Four hash seeds now agree to six decimals. **If you are comparing against
numbers from an older revision of this file, they are one draw from that
lottery — re-run rather than reconciling.** All Task 2 tables above have been
re-measured; the conclusions held, with the margin in finding 3 tightened.

#### 1. `match_id` dtype is inconsistent — silent empty joins

This is the one that actually bites, because it fails with **zero rows and no
error**.

| dtype | Files |
|---|---|
| **`str`** | `corners.parquet`, `corner_deliveries.parquet`, `defense_features.parquet` |
| **`int64`** | `corner_delivery_full.parquet`, `corners_full_raw.parquet`, and **every** `data/processed/*.parquet` |

Merging a subset file on `match_id` against a pipeline file returns nothing.
Cast first: `df.match_id.astype(int)`. Joining on `corner_id` / `uuid` is
unaffected — those are `str` everywhere.

#### 2. The corner key has two different names

Same values, two names, depending on which side of the project the file came
from:

- **`corner_id`** — every `tacticai_task2/*.parquet` except one
- **`uuid`** — `corners_full_raw.parquet` (it came from `decisions.parquet`),
  and all `data/processed/*.parquet`

`corners_full_raw.parquet` is the trap: it lives in `tacticai_task2/` but uses
the pipeline's key name. Rename on load:
`.rename(columns={'uuid': 'corner_id'})`.

#### 3. Three near-identical delivery filenames

| File | Rows | Scope |
|---|---:|---|
| `corner_delivery.parquet` | 2,473 | 360 subset — technique / height / body part only |
| `corner_deliver**ies**.parquet` | 2,473 | 360 subset — **plus** reflected endpoints, flight time, contact height. Built and cached by `synthetic_trajectories.py` |
| `corner_delivery_**full**.parquet` | 41,147 | Full dataset |

Singular vs plural is a one-letter difference carrying a real schema
difference. `corner_deliveries.parquet` is the one the recommender and the
defence/delivery analyses read.

#### 4. Scope: 2,473 vs 41,178 corners

Files with parallel names cover different match sets. Anything with freeze
frames is the **2,473**-corner 360 subset; anything `*_full*` is all **41,178**
corners with no 360 data. `defense_features.parquet` is smaller still (**2,262**)
because it keeps only long corners. Row counts are the quickest way to tell what
you have.

#### 5. Orphaned `.csv` twins — will silently go stale

`corners.csv`, `corner_players.csv` and `player_bio.csv` are byte-parallel to
their `.parquet` versions **and no script writes them.** They were dumped by an
older version of the extractors. They match today, but re-running
`extract_corners.py` or `fetch_bio.py` updates only the parquet, leaving the csv
quietly stale. **Treat the parquet as authoritative.**

The reorg makes these easier to spot rather than harder: they are now the
**only** `.csv` files left in the script directory, since everything a script
writes moved to `results/`. That is also why they were not moved — putting them
under `results/` would assert they are generated output, which is exactly the
false claim that makes them dangerous. A `.csv` beside the `.py` files is an
orphan by construction.

#### 6. A superseded script with the most authoritative-looking name

`tactic_suggest.py` reads like the main entry point and is not — see the
recommendation lineage above. `tactic_recommend.py` is the live one.

### Output collisions

Three filenames are written by more than one script, across all 42 distinct
outputs in the project. In every case the pair is a script and its own later
iteration, so the collision only bites if you run the **older** one after the
newer — which would silently replace current results with superseded ones.

| Filename | Written by | Currently on disk from |
|---|---|---|
| `results/tactic_suggestions.csv` | `tactic_suggest.py`, `tactic_suggest2.py` | **v2** — header is `corner_id, shot_real, p_original, p_attack_1, p_attack_bestk, p_defense_1, p_defense_bestk`; the `_bestk` columns are v2's, and v1 writes `p_attack_adj` / `p_defense_adj` instead |
| `figures/tactic_suggestions.png` | `tactic_suggest.py`, `tactic_suggest2.py` | **v2** — same mtime as the csv above |
| `results/ablation_batch.csv` | `ablate_batch.py`, `finish_ablation.py` | **`finish_ablation.py`** — it appended the batch-512 row to the three `ablate_batch.py` had written, so the file is the union and neither script alone reproduces it |

Ownership is decidable from the file contents in each case, so nothing on disk
is actually ambiguous — but running `tactic_suggest.py` now would overwrite v2's
results with v1's under the same names and lose that.

**Separately, a self-overwrite risk.** `diffusion_corners.py` writes
`diffusion_model{tag}.pt` and `diffusion_loss{tag}.csv`, where `tag` comes from
`--tag`. Re-running with a tag that already exists silently replaces a
checkpoint that took ~45 minutes to train, with no warning. The five committed
checkpoints (`diffusion_model.pt` plus `_b64/_b128/_b256/_b512`) are each the
product of one such run.

`tactic_suggest3.py` and `tactic_recommend.py` write under distinct names
(`tactic_joint.*` and `tactic_recommendation*`), so the two most recent
iterations do not collide with anything.

### Data files

| File | What it is |
|---|---|
| `corners.parquet` / `.csv` | One row per subset corner: match, labels (`shot`, receiver metadata), flight dt, velocity coverage, `reflected` flag |
| `corner_players.parquet` / `.csv` | One row per visible freeze-frame player: normalised position, flags, velocity, `is_receiver` |
| `corners_full_raw.parquet` | All 41,178 corners with endpoints and outcome-forward labels (no freeze frames) |
| `corner_delivery.parquet` | Delivery attributes, 2,473 subset corners |
| `corner_delivery_full.parquet` | Delivery attributes, 41,147 full-set corners |
| `corner_deliveries.parquet` | Subset corners with reflected endpoints, flight time and contact height, ready for trajectory reconstruction |
| `onpitch.parquet` | Per corner: attacking/defending player-id lists and GK ids |
| `player_bio.parquet` / `.csv` | Per player: name, Wikidata QID, height_cm, weight_kg (unit-cleaned) |
| `models/diffusion_model.pt` | Trained EMA checkpoint (1.09M params, 60k steps) |
| `models/diffusion_model_b{64,128,256,512}.pt` | Batch-ablation checkpoints; load via `load_model(tag="_b128")` |
| `results/diffusion_split.csv` | Train/holdout template split used by every diffusion experiment |
| `results/diffusion_loss.csv`, `results/diffusion_loss_b*.csv` | Training loss curves |
| `synthetic_corners.parquet` | Generated corners, 3 variants per held-out template (real label, shot=1, shot=0) |
| `results/results_seeds.csv` / `results/results_summary.csv` | Per-seed and aggregated Task 2 metrics per configuration |
| `tactic_best_players.parquet` | Chosen player coordinates from the v2 best-of-k loop |
## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python extract_corners.py   # ~2 min, needs ~/Desktop/open-data-master
.venv/bin/python onpitch.py
.venv/bin/python fetch_bio.py         # ~30 min, hits the Wikidata API
.venv/bin/python train_eval.py        # 9 configs x 5 seeds, ~35 min
.venv/bin/python diffusion_corners.py # train diffusion (needs torch; ~45 min on M3 Pro)
.venv/bin/python diffusion_eval.py    # sample + evaluate + figure
```

(`requirements.txt` covers the tabular pipeline; the diffusion scripts
additionally need `torch` and `matplotlib`.)

Two checks worth running after any change to paths or to the receiver model:

```bash
.venv/bin/python check_paths.py --strict      # no write escapes results/figures/models
.venv/bin/python train_eval.py --check-cancel # globals reach the receiver only via tanh
```

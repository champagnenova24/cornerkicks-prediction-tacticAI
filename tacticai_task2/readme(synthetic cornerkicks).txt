====================================================================
SYNTHETIC CORNER KICK GENERATION WITH A DIFFUSION MODEL
====================================================================

Reproduces the generative component (Task 3) of "TacticAI: an AI
assistant for football tactics" (Wang et al., Nature Communications
15:1906, 2024) with a diffusion model instead of the paper's
conditional VAE, trained on StatsBomb open data 360 freeze frames.

The paper's test: generate corner-kick player setups and check that
(a) an MLP classifier cannot tell them apart from real corners
    (they report F1 = 0.53 +/- 0.05, i.e. chance level), and
(b) the setups can be steered toward or away from shot-producing
    configurations.

--------------------------------------------------------------------
1. WHAT THE MODEL GENERATES
--------------------------------------------------------------------
One corner kick = a set of players, each with:
  - position (x, y)      [pitch coords, corner normalized to (120, 0)]
  - velocity (vx, vy)    [flight-window estimate, see main README.md]

Fixed per-player conditioning (taken from a real corner's "roster
template", not generated):
  - teammate flag (attacker / defender)
  - keeper flag
  - actor flag (the corner taker)
  - has_vel flag (was a velocity observed for this player)

Global conditioning:
  - outcome label: shot / no-shot / null (classifier-free guidance)

This mirrors TacticAI's setup where a fixed set of players is
repositioned; the roster comes from data, the placement is generated.

--------------------------------------------------------------------
2. MODEL AND TRAINING
--------------------------------------------------------------------
Denoiser      : permutation-equivariant transformer over player slots
                (no positional encoding across players), d=160,
                5 layers, 4 heads, 1.09M parameters.
Diffusion     : DDPM, 1000 steps, cosine noise schedule, epsilon
                prediction, EMA of weights (0.9995).
Conditioning  : label embedding added to every token; label dropped
                to "null" 15% of training steps (classifier-free
                guidance). Sampling uses DDIM (200 steps) with
                guidance weight w:
                  w = 1  -> realism sampling (plain conditional)
                  w = 2  -> control sampling (push toward/away shot)
Loss masking  : positions trained on all visible players; velocity
                loss only where a velocity was actually observed.
Augmentation  : 0.25 m Gaussian position jitter.
Data          : 2,425 corners (>=5 visible players per team),
                1,940 train / 485 holdout (diffusion_split.csv).
Training      : 60,000 steps, batch 256, AdamW, cosine LR decay.

LAPTOP FEASIBILITY (measured on an Apple M3 Pro, PyTorch MPS):
  - the script benchmarks itself before training and ABORTS if the
    projected time exceeds 90 minutes (TIME_BUDGET_MIN);
  - measured: ~23 steps/s -> 60k steps in 44 minutes (a smaller
    0.56M model trained in 12 minutes);
  - sampling 3 x 485 corners at 200 DDIM steps: ~2 minutes.
  Conclusion: a laptop is sufficient for this task. Bigger hardware
  only becomes relevant for hyperparameter sweeps or full
  trajectory-level diffusion.

--------------------------------------------------------------------
3. RESULTS (485 held-out roster templates)
--------------------------------------------------------------------
Realism -- can a classifier tell real from synthetic?
(0.5 = indistinguishable; paper's CVAE: MLP F1 0.53)

  discriminator            F1      AUC
  MLP (paper's test)       0.62    0.70
  HistGradientBoosting     0.69    0.74   (stronger, stricter test)

Guidance -- mean P(shot) assigned by an independent shot classifier
trained on real corners only:

  real corners, no shot        0.164
  generated, guided shot=0     0.182
  generated, real labels       0.202
  generated, guided shot=1     0.248
  real corners, shot           0.269

Reading:
  + Outcome steering works: shot=1 vs shot=0 guidance recovers ~78%
    of the real shot/no-shot separation (the TacticAI "adjust players
    to change shot probability" capability). The effect holds on
    average; individual templates are noisy (see gallery, row 2).
  - Realism is close but not at the paper's chance level: synthetic
    corners are slightly under-packed in the goalmouth (fewer players
    in the 6-yard box, larger min distance to goal; roughly 0.3 sd
    gaps -- see diffusion_distributions.csv). This is typical
    diffusion over-smoothing of dense clusters on small data.

Why the paper's realism number is better: ~3x more corners, complete
22-player pre-kick tracking (our velocities are flight-window
estimates over ~75% of visible players), and their MLP judged raw
inputs while our discriminators see engineered aggregate features --
a stricter test. Note: doubling model size and training time did NOT
improve realism, so the next lever is method (v-prediction / EDM
preconditioning, min-SNR weighting, self-conditioning, flow matching)
or data, not compute.

--------------------------------------------------------------------
3c. EXTENDED QUALITY METRICS (quality_metrics.py)
--------------------------------------------------------------------
Five axes that fail independently, each with a REAL-vs-REAL control
(holdout split in half) showing what a perfect generator would score.
Descriptor = histograms of att-att / def-def / att-def pairwise
distances (permutation-invariant, no engineered features).

  metric                        synthetic     control (real/real)
  MMD^2 (RBF) + perm p          0.0056 p.001  -0.0005 p=0.62
    att-att / def-def / att-def 0.0029 / 0.0066 / 0.0119
  1-NN two-sample accuracy      0.540         0.487
  precision (fidelity)          0.829         0.811
  recall (diversity)            0.810         0.868
  density                       1.057         0.840
  coverage                      0.959         0.967
  median Chamfer to nearest
    TRAINING corner             2.04 m        1.99 m (real holdout)

Physical plausibility (same checks run on real corners):

  check                         real        synthetic
  players beyond goal line      0.5%        2.4%
  players outside side lines    0.08%       3.0%
  corners w/ <0.5 m overlap     32%         38%
  corners w/ >9 m/s player      5.4%        2.9%

CONCLUSIONS
  - No memorization: nearest-training-corner distance ratio 1.02.
  - No mode collapse: coverage 0.96, fidelity equals the control;
    diversity is mildly reduced (recall 0.81 vs 0.87, density > 1).
  - The distribution gap is real (MMD p = 0.001) and is concentrated
    in ATTACKER-DEFENDER structure (largest per-type MMD). The g(r)
    curves show why: at r < 2 m real corners have far more
    attacker-defender pairs (0.0062 vs 0.0043) -- the model softens
    tight man-marking contact while slightly over-clustering
    same-team players.
  - Boundary softness: overshoots are sub-meter on average (max 2 m),
    a diffusion artifact at hard constraints. Fix by clamping x/y to
    the pitch during sampling (the current +-2.5 normalized clamp is
    far too loose to bound the pitch).
  - 1-NN (0.54) is much closer to chance than the feature-based
    discriminator (F1 0.62): the discriminator's advantage comes from
    engineered global features (box counts, velocity aggregates),
    not from relational structure.

Outputs: quality_metrics.csv, quality_pcf.csv, quality_metrics.png
(g(r) curves per pair type + nearest-training-corner histogram).

--------------------------------------------------------------------
3z. POSITIONING SUGGESTIONS (PAPER FIG. 3 ANALOG)
--------------------------------------------------------------------
Coaching adjustments by INPAINTING: freeze one team at its real
positions, re-denoise the other team from noise under the desired
outcome label (tactic_suggest.py / tactic_suggest2.py).

  ATTACK  : defenders + GK + taker frozen, outfield attackers
            regenerated with label shot = 1  (guidance w = 2)
  DEFENSE : attackers + taker frozen, outfield defenders
            regenerated with label shot = 0  (guidance w = 2)

Positions AND velocities are regenerated, so each suggestion says
"stand here, moving this way". tactic_suggest2.py adds the paper's
propose-and-evaluate loop: 6 candidates per corner, the independent
shot predictor keeps the best (max P attack / min P defense).

Mean P(shot) over 485 holdout corners:

  original                        0.185
  ATTACK  single draw             0.220
  ATTACK  best-of-6               0.323   (improves 94% of corners)
  DEFENSE single draw             0.204
  DEFENSE best-of-6               0.142   (improves 82% of corners)

On the 100 holdout corners with a real shot (the paper's evaluation
slice): defense best-of-6 cuts P(shot) 0.269 -> 0.198.
Paper reference point: 0.75 -> 0.69 on their 100 threatening corners
(absolute levels differ by dataset/predictor; direction and relative
size are the comparable part).

A SINGLE conditional draw is not reliably an improvement for defense
(0.204 > 0.185): w = 2 guidance moves players tactically but also
off-distribution. Generating several candidates and letting the shot
predictor choose is what makes it work -- the same predictive +
generative interplay the paper describes.

Outputs: tactic_suggestions.png (original / attack / defense panels
with original->suggested arrows), tactic_suggestions.csv (per corner),
tactic_best_players.parquet (chosen player coordinates).

--------------------------------------------------------------------
3a. TSTR (TRAIN ON SYNTHETIC, TEST ON REAL)
--------------------------------------------------------------------
One synthetic corner per real training template (w=1, labels = the
conditioning labels); the Task-2 shot classifier is trained on each
source and evaluated on the same 485 REAL holdout corners
(tstr_eval.py -> tstr_results.csv; mean +- std over 3 seeds):

  source                        n      AUC              F1
  TRTR real corners             1940   0.719 +- 0.010   0.447 +- 0.010
  TSTR synthetic only           1940   0.693 +- 0.008   0.408 +- 0.007
  AUG  real + synthetic         3880   0.696 +- 0.007   0.419 +- 0.008

A classifier that never saw a real corner keeps ~88% of the real
model's above-chance signal -> the generator captures most of the
decision-relevant structure, despite being detectable by the
discriminators. Augmentation does not help: same templates twice
adds smoothing, not information.

--------------------------------------------------------------------
3b. BATCH-SIZE ABLATION
--------------------------------------------------------------------
Same architecture/split, 30k steps, batch in {64, 128, 256, 512}
(ablate_batch.py -> ablation_batch.csv; one seed per setting):

  batch   MLP F1/AUC   HGB F1/AUC   P(shot): shot0 -> shot1
  64      0.65 / 0.71  0.65 / 0.74  0.18 -> 0.25
  128     0.60 / 0.68  0.62 / 0.69  0.18 -> 0.25
  256     0.65 / 0.69  0.66 / 0.73  0.18 -> 0.25
  512     0.57 / 0.62  0.66 / 0.72  0.18 -> 0.25

Guidance separation is flat across batch sizes. Realism has no
consistent winner (MLP favors 512, the stronger HGB favors 128;
64 is worst for both; single-seed differences < ~0.04 are noise).
Batch size is not the lever that fixes the realism gap. Ablation
checkpoints: diffusion_model_b{64,128,256,512}.pt, loadable via
load_model(tag="_b128") etc. Long runs on a laptop: prevent sleep
with `caffeinate -is <command>` or training silently stalls.

--------------------------------------------------------------------
4. FILES
--------------------------------------------------------------------
diffusion_corners.py        model + training (with the feasibility
                            gate) + DDIM sampler; CLI:
                            --batch N --steps N --tag NAME
ablate_batch.py             batch-size ablation driver
ablation_batch.csv          ablation results table
finish_ablation.py          re-evaluate one checkpoint and merge into
                            ablation_batch.csv
diffusion_eval.py           sampling, discriminator realism test,
                            guidance check, distribution table, figure
gallery.py                  example-figure generator
diffusion_model.pt          trained EMA checkpoint (1.09M params)
diffusion_split.csv         train/holdout template split
diffusion_loss.csv          training loss curve
diffusion_eval.csv          all metrics above
diffusion_distributions.csv real vs synthetic feature means/stds
synthetic_corners.parquet   generated corners, 3 variants per
                            holdout template (real label / shot=1 /
                            shot=0), corner_id suffixed accordingly
real_vs_synthetic.png       3 held-out corners, real vs generated
synthetic_gallery.png       gallery: real vs guided generations with
                            independent shot-probability annotations,
                            plus a sample-diversity row

Upstream inputs (built by the Task 2 pipeline, see README.md):
corners.parquet, corner_players.parquet.

--------------------------------------------------------------------
5. HOW TO RUN
--------------------------------------------------------------------
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt torch matplotlib
.venv/bin/python extract_corners.py      # if parquets not present
.venv/bin/python diffusion_corners.py    # train  (~45 min on M3 Pro)
.venv/bin/python diffusion_eval.py       # sample + evaluate + figure
.venv/bin/python gallery.py              # example gallery

Reading synthetic_corners.parquet: one row per generated player
(corner_id, variant, x, y, vx, vy, teammate, keeper, actor, has_vel).
Velocities are zeroed where the template had has_vel = 0, matching
how real corners are stored.
====================================================================

"""A/B test: does constraining the sampler to the pitch improve realism?

Paired comparison on the SAME checkpoint and the SAME sampling seeds, so the
only difference between arms is `Diffusion.project_x0`:

  off : x0 clamped to +-2.5 in normalized units (the original behaviour)
  on  : position channels additionally projected onto the pitch + the real
        out-of-bounds margin

Reported per arm:
  1. real-vs-synthetic discriminators (MLP + HGB) over N random splits, so a
     difference can be read against the seed noise rather than a single number
  2. constraint violations, per player and per corner
  3. goalmouth occupancy (the under-densification the ablations kept hitting)
  4. guidance separation, to confirm the projection does not cost control

Writes ablation_projection.csv.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from diffusion_corners import SP, load_dataset, load_model
from diffusion_eval import (sample_variants, to_player_frame,
                            real_player_frame, globals_for)
from train_eval import CONFIGS

N_DISC_SEEDS = 10

# (label, project, margin_yd). The margin matters: at 3.25 yd the bound admits
# every excursion the real data shows and so almost never binds; at 0.0 it is
# the strict pitch and drives out-of-bounds to zero, at the cost of being more
# constrained than the real freeze frames actually are.
ARMS = [
    ('off (baseline)', False, 0.0),
    ('on, margin 3.25yd', True, 3.25),
    ('on, strict pitch', True, 0.0),
]
BASELINE = ARMS[0][0]


def discriminators(G_real, G_syn, n_seeds=N_DISC_SEEDS):
    """Real vs synthetic, over n_seeds random splits. Returns per-seed arrays."""
    Xd = np.vstack([G_real.to_numpy(float), G_syn.to_numpy(float)])
    yd = np.concatenate([np.ones(len(G_real)), np.zeros(len(G_syn))])
    res = {k: [] for k in ['mlp_f1', 'mlp_auc', 'hgb_f1', 'hgb_auc']}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xd))
        n_tr = int(0.7 * len(Xd))
        tr, te = perm[:n_tr], perm[n_tr:]
        sc = StandardScaler().fit(Xd[tr])
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                            random_state=seed).fit(sc.transform(Xd[tr]), yd[tr])
        p = mlp.predict_proba(sc.transform(Xd[te]))[:, 1]
        res['mlp_f1'].append(f1_score(yd[te], p >= 0.5))
        res['mlp_auc'].append(roc_auc_score(yd[te], p))
        hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xd[tr], yd[tr])
        p = hgb.predict_proba(Xd[te])[:, 1]
        res['hgb_f1'].append(f1_score(yd[te], p >= 0.5))
        res['hgb_auc'].append(roc_auc_score(yd[te], p))
    return {k: np.array(v) for k, v in res.items()}


def constraint_stats(dp):
    """Out-of-pitch rates, per player and per corner, plus goalmouth share."""
    x, y = dp.x.to_numpy(), dp.y.to_numpy()
    oob = (x < 0) | (x > 120) | (y < 0) | (y > 80)
    excess = np.maximum.reduce([np.zeros_like(x), x - 120, -x, y - 80, -y])
    per_corner = dp.assign(_oob=oob).groupby('corner_id')._oob.any()
    return {
        'oob_player_pct': 100 * oob.mean(),
        'oob_corner_pct': 100 * per_corner.mean(),
        'oob_median_excess_yd': (float(np.median(excess[oob])) if oob.any()
                                 else 0.0),
        'behind_goal_player_pct': 100 * (x > 120).mean(),
        'in_6yd_box_pct': 100 * ((x >= 114) & (y >= 30) & (y <= 50)).mean(),
        'in_18yd_box_pct': 100 * ((x >= 102) & (y >= 18) & (y <= 62)).mean(),
    }


def main(device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f'{SP}/diffusion_split.csv')
    hold_mask = split.split.to_numpy() == 'holdout'
    hold_idx, train_idx = np.where(hold_mask)[0], np.where(~hold_mask)[0]
    model, diff = load_model(device)

    gcols = CONFIGS['vel'][1]
    dp_real_hold = real_player_frame(ids, X, F, M, hold_idx)
    dp_real_train = real_player_frame(ids, X, F, M, train_idx)
    G_real_hold = globals_for(dp_real_hold, gcols)
    G_real_train = globals_for(dp_real_train, gcols)

    # independent shot classifier for the guidance check (trained on real only)
    shot_by_cid = dict(zip(ids, y))
    y_tr = np.array([shot_by_cid[c] for c in G_real_train.index])
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_real_train.to_numpy(float), y_tr)

    print(f'{len(hold_idx)} holdout templates, {N_DISC_SEEDS} discriminator '
          f'seeds per arm, device {device}\n', flush=True)

    rows, disc_raw = [], {}
    for arm, project, margin in ARMS:
        print(f'--- sampling arm: {arm} ---', flush=True)
        variants = sample_variants(model, diff, F, M, y, hold_idx, device,
                                   project=project, margin_yd=margin)
        dp_syn = {k: to_player_frame(ids, v, F, M, hold_idx, k)
                  for k, v in variants.items()}
        G_syn = {k: globals_for(v, gcols) for k, v in dp_syn.items()}

        d = discriminators(G_real_hold, G_syn['real_label'])
        disc_raw[arm] = d
        row = {'arm': arm}
        for k, v in d.items():
            row[f'{k}_mean'], row[f'{k}_std'] = v.mean(), v.std()
        row.update(constraint_stats(dp_syn['real_label']))
        row['p_shot_shot1'] = float(
            clf.predict_proba(G_syn['shot1'].to_numpy(float))[:, 1].mean())
        row['p_shot_shot0'] = float(
            clf.predict_proba(G_syn['shot0'].to_numpy(float))[:, 1].mean())
        row['guidance_gap'] = row['p_shot_shot1'] - row['p_shot_shot0']
        rows.append(row)

    real_row = {'arm': 'REAL holdout'}
    real_row.update(constraint_stats(dp_real_hold))
    res = pd.DataFrame(rows + [real_row])
    res.to_csv(f'{SP}/ablation_projection.csv', index=False)

    def show(cols, title, fmt='{:.3f}'):
        print(f'\n=== {title} ===')
        sub = res.set_index('arm')[cols].dropna(how='all')
        print(sub.map(lambda v: fmt.format(v) if pd.notna(v) else '-')
              .to_string())

    show(['mlp_f1_mean', 'mlp_f1_std', 'mlp_auc_mean', 'mlp_auc_std',
          'hgb_f1_mean', 'hgb_f1_std', 'hgb_auc_mean', 'hgb_auc_std'],
         f'1. discriminator (lower = more realistic), {N_DISC_SEEDS} seeds')
    show(['oob_player_pct', 'oob_corner_pct', 'oob_median_excess_yd',
          'behind_goal_player_pct'], '2. constraint violations', '{:.2f}')
    show(['in_6yd_box_pct', 'in_18yd_box_pct'], '3. goalmouth occupancy',
         '{:.2f}')
    show(['p_shot_shot1', 'p_shot_shot0', 'guidance_gap'],
         '4. guidance separation (should be unchanged)')

    # paired significance on the discriminator metrics
    print(f'\n=== 5. arm difference vs seed noise (paired, vs baseline) ===')
    a = disc_raw[BASELINE]
    for arm in disc_raw:
        if arm == BASELINE:
            continue
        print(f'  {arm}:')
        for k in a:
            d = disc_raw[arm][k] - a[k]
            verdict = 'CHANGED' if abs(d.mean()) > 2 * d.std() else 'within noise'
            print(f'    {k:>8}: {d.mean():+.4f}  '
                  f'(per-seed sd {d.std():.4f}) -> {verdict}')
    print('\nwrote ablation_projection.csv')


if __name__ == '__main__':
    main()

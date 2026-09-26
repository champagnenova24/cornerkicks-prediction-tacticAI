"""Does conditioning the generator on the delivery fix what it was missing?

`synthetic_trajectories.py` diagnosed the delivery-blind generator: on the same
deliveries, real setups put an attacker within 3 yd of the arrival point on
73% of corners and generated ones on 34%, and the 6-yd box holds 28.7% of
generated players against 35.1% real. Its prediction was that one missing
input -- the delivery -- causes both. `diffusion_delivery.py` supplies it.
This script tests the prediction.

Arms, all on the 485 holdout templates of `results/diffusion_split.csv`:

  baseline       the committed delivery-blind model (1.09M, 60k steps), read
                 from synthetic_corners.parquet -- exactly the numbers in the
                 README
  blind          the NEW checkpoint with the delivery dropped. Same weights and
                 training budget as `delivery`, so `delivery - blind` is the
                 effect of the conditioning alone
  delivery       the new checkpoint, told the template's real delivery
  shuffled       the new checkpoint, told ANOTHER holdout corner's delivery.
                 If the model uses the ball, players follow the ball they were
                 given, not the one the template really had
  shot1 / shot0  real delivery, label guidance w = 2 on top of it

Metrics: ball consistency (as synthetic_trajectories.py), goalmouth occupancy
and out-of-bounds (as ablate_projection.py), real-vs-synthetic discriminators
over 10 splits, and guidance separation. Then a query figure: one roster, the
generator asked for different deliveries.

Outputs: results/delivery_eval.csv, results/delivery_ball_consistency.csv,
         figures/delivery_conditioning.png, synthetic_corners_delivery.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from diffusion_corners import SP, load_dataset, pick_device
from diffusion_delivery import (load_model, load_deliveries, encode_delivery,
                                to_yards)
from diffusion_eval import to_player_frame, real_player_frame, globals_for
from ablate_projection import discriminators, constraint_stats
from synthetic_trajectories import consistency, draw
from train_eval import CONFIGS

sys.path.insert(0, os.path.join(os.path.dirname(SP), 'pipeline'))
from ball_trajectory import reconstruct  # noqa: E402

DDIM_STEPS = 200
W_REAL, W_CTRL = 1.0, 2.0
LONG_CHORD = 15.0


def sample(model, diff, F, M, labels, D, idx, guidance, device, seed0=1234):
    """D=None samples delivery-blind. Same seeds as diffusion_eval."""
    dev = torch.device(device)
    chunks = []
    for s in range(0, len(idx), 128):
        sl = idx[s:s + 128]
        z = diff.ddim_sample(
            model, torch.tensor(F[sl]).to(dev), torch.tensor(M[sl]).to(dev),
            torch.tensor(labels[s:s + 128]).to(dev),
            None if D is None else torch.tensor(D[s:s + 128]).to(dev),
            steps=DDIM_STEPS, guidance=guidance, seed=seed0 + s)
        chunks.append(z.cpu().numpy())
    return to_yards(np.concatenate(chunks), COORDS)


def summarize_consistency(c):
    return pd.Series({
        'n': len(c),
        'att_kick_median': c.d_att_kick.median(),
        'pct_att_within_3yd_kick': 100 * c.att_near_kick.mean(),
        'att_arrival_median': c.d_att_arrival.median(),
        'pct_att_within_3yd_arrival': 100 * c.att_near_arrival.mean(),
    })


COORDS = 'linear'      # set from the checkpoint in main()


def main(device=None, tag=''):
    """`tag` picks the checkpoint (models/diffusion_delivery{tag}.pt) and
    suffixes every output, so the default reproduces the committed files."""
    global COORDS
    device = device or pick_device()
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f'{SP}/results/diffusion_split.csv')
    hold = np.where(split.split.to_numpy() == 'holdout')[0]
    train = np.where(split.split.to_numpy() == 'train')[0]
    model, diff = load_model(device, tag)
    COORDS = getattr(model, 'coords', 'linear')
    print(f'checkpoint diffusion_delivery{tag}.pt, coordinates: {COORDS}')
    D = load_deliveries(ids, COORDS)
    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet')

    # a fixed derangement of the holdout deliveries, restricted to long ones so
    # a shuffled ball is still a real cross into the box
    hold_ids = [ids[i] for i in hold]
    dlh = dl.set_index('corner_id').loc[hold_ids]
    chord = np.hypot(dlh.end_x - dlh.start_x, dlh.end_y - dlh.start_y).to_numpy()
    long_pos = np.where(chord >= LONG_CHORD)[0]
    rng = np.random.default_rng(0)
    perm = np.arange(len(hold))
    while True:
        p = rng.permutation(long_pos)
        if not (p == long_pos).any():
            break
    perm[long_pos] = p
    D_shuf = D[hold][perm]
    dl_shuf = dlh.iloc[perm].reset_index(drop=True).assign(
        corner_id=hold_ids)            # template id, OTHER corner's ball

    print(f'sampling {len(hold)} holdout templates, {DDIM_STEPS} DDIM steps',
          flush=True)
    Z = {
        'blind': sample(model, diff, F, M, y[hold], None, hold, W_REAL, device),
        'delivery': sample(model, diff, F, M, y[hold], D[hold], hold, W_REAL, device),
        'shuffled': sample(model, diff, F, M, y[hold], D_shuf, hold, W_REAL, device),
        'shot1': sample(model, diff, F, M, np.ones(len(hold), np.int64),
                        D[hold], hold, W_CTRL, device),
        'shot0': sample(model, diff, F, M, np.zeros(len(hold), np.int64),
                        D[hold], hold, W_CTRL, device),
    }
    frames = {k: to_player_frame(ids, v, F, M, hold, k) for k, v in Z.items()}
    base = pd.read_parquet(f'{SP}/synthetic_corners.parquet')
    frames['baseline'] = base[base.variant == 'real_label'].drop(columns='variant')
    real = real_player_frame(ids, X, F, M, hold)
    pd.concat([v.assign(variant=k) for k, v in frames.items()
               if k != 'baseline']) \
        .to_parquet(f'{SP}/synthetic_corners_delivery{tag}.parquet', index=False)

    # ---- 1. ball consistency --------------------------------------------
    dl_h = dl[dl.corner_id.isin(hold_ids)]
    cons = {'real': consistency(real, dl_h, 'real')}
    for k in ['baseline', 'blind', 'delivery', 'shot1', 'shot0']:
        cons[k] = consistency(frames[k], dl_h, k)
    # shuffled: scored against the ball it was GIVEN, and against the
    # template's own ball it was not told about
    cons['shuffled (given ball)'] = consistency(frames['shuffled'], dl_shuf,
                                                'shuffled_given')
    cons['shuffled (own ball)'] = consistency(frames['shuffled'], dl_h,
                                              'shuffled_own')
    long_ids = set(np.array(hold_ids)[long_pos])
    tab, tab_long = {}, {}
    for k, c in cons.items():
        base_id = c.corner_id.str.split('__').str[0]
        tab[k] = summarize_consistency(c)
        tab_long[k] = summarize_consistency(c[base_id.isin(long_ids)])
    tab, tab_long = pd.DataFrame(tab).T, pd.DataFrame(tab_long).T
    print('\n=== 1. ball consistency: nearest attacker to the arrival point ===')
    print('all holdout corners')
    print(tab.round(2).to_string())
    print(f'\nlong deliveries only (chord >= {LONG_CHORD:.0f} yd), '
          'where the shuffle applies')
    print(tab_long.round(2).to_string())
    pd.concat([tab.assign(subset='all'), tab_long.assign(subset='long')]) \
        .rename_axis('arm').reset_index() \
        .to_csv(f'{SP}/results/delivery_ball_consistency{tag}.csv', index=False)

    # ---- 2. goalmouth + realism + guidance ------------------------------
    gcols = CONFIGS['vel'][1]
    G_real = globals_for(real, gcols)
    G_train = globals_for(real_player_frame(ids, X, F, M, train), gcols)
    shot_by = dict(zip(ids, y))
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_train.to_numpy(float), [shot_by[c] for c in G_train.index])
    G = {k: globals_for(frames[k], gcols) for k in
         ['baseline', 'blind', 'delivery', 'shuffled', 'shot1', 'shot0']}

    rows = []
    for k in ['baseline', 'blind', 'delivery', 'shuffled']:
        r = {'arm': k}
        d = discriminators(G_real, G[k])
        for m, v in d.items():
            r[f'{m}_mean'], r[f'{m}_std'] = v.mean(), v.std()
        r.update(constraint_stats(frames[k]))
        att = frames[k][frames[k].teammate & ~frames[k].actor]
        r['att_in_6yd_pct'] = 100 * ((att.x >= 114) & (att.y >= 30)
                                     & (att.y <= 50)).mean()
        r.update(tab.loc[k if k != 'shuffled' else 'shuffled (given ball)',
                         ['pct_att_within_3yd_kick',
                          'pct_att_within_3yd_arrival']].to_dict())
        rows.append(r)
    rr = {'arm': 'REAL holdout'}
    rr.update(constraint_stats(real))
    att = real[real.teammate & ~real.actor]
    rr['att_in_6yd_pct'] = 100 * ((att.x >= 114) & (att.y >= 30)
                                  & (att.y <= 50)).mean()
    rr.update(tab.loc['real', ['pct_att_within_3yd_kick',
                                'pct_att_within_3yd_arrival']].to_dict())
    rows.append(rr)
    res = pd.DataFrame(rows).set_index('arm')
    p1 = float(clf.predict_proba(G['shot1'].to_numpy(float))[:, 1].mean())
    p0 = float(clf.predict_proba(G['shot0'].to_numpy(float))[:, 1].mean())
    res.loc['delivery', 'p_shot_shot1'] = p1
    res.loc['delivery', 'p_shot_shot0'] = p0
    res.loc['delivery', 'guidance_gap'] = p1 - p0
    res.to_csv(f'{SP}/results/delivery_eval{tag}.csv')

    print('\n=== 2. goalmouth occupancy (% of players; attackers for att_) ===')
    print(res[['in_6yd_box_pct', 'att_in_6yd_pct', 'in_18yd_box_pct',
               'oob_player_pct']].round(1).to_string())
    print('\n=== 3. realism: real vs synthetic discriminators, 10 splits ===')
    print(res[['mlp_f1_mean', 'mlp_auc_mean', 'hgb_f1_mean', 'hgb_auc_mean',
               'hgb_auc_std']].dropna().round(3).to_string())
    print(f'\n=== 4. label guidance on top of the real delivery (w={W_CTRL}) ===')
    print(f'  P(shot) guided to shot=1 {p1:.3f}, to shot=0 {p0:.3f}, '
          f'gap {p1 - p0:+.3f}  (delivery-blind model: 0.248 / 0.182, +0.066)')

    # ---- 5. query figure ------------------------------------------------
    query_figure(model, diff, ids, F, M, y, hold, dl, device, tag)
    print(f'\nwrote results/delivery_eval{tag}.csv, results/delivery_ball_consistency{tag}.csv,'
          f' figures/delivery_conditioning{tag}.png, synthetic_corners_delivery{tag}.parquet')


QUERIES = [
    ('inswinger to the near post', 116.5, 36.0, 'Inswinging', 'High Pass'),
    ('inswinger, central 6-yd box', 115.5, 40.5, 'Inswinging', 'High Pass'),
    ('outswinger to the far post', 111.5, 47.0, 'Outswinging', 'High Pass'),
]


def query_figure(model, diff, ids, F, M, y, hold, dl, device, tag='', n_draw=3):
    """One roster, three delivery queries, n_draw samples each; plus the
    average response of the whole holdout set to the same queries."""
    dev = torch.device(device)
    dur = dl.groupby('technique').duration.median()
    # the template: a long, high holdout delivery with a full roster
    dlh = dl.set_index('corner_id')
    cands = [i for i in hold
             if dlh.loc[ids[i], 'height_class'] == 'High Pass'
             and M[i].sum() >= 18]
    ti = sorted(cands, key=lambda i: ids[i])[0]
    fig, axes = plt.subplots(n_draw + 1, len(QUERIES),
                             figsize=(3.6 * len(QUERIES), 3.3 * (n_draw + 1)))
    for col, (name, ex, ey, tech, height) in enumerate(QUERIES):
        dv = encode_delivery(ex, ey, tech, height, COORDS)
        rep = np.repeat([ti], n_draw)
        z = diff.ddim_sample(
            model, torch.tensor(F[rep]).to(dev), torch.tensor(M[rep]).to(dev),
            torch.full((n_draw,), 2, dtype=torch.long, device=dev),
            torch.tensor(np.repeat(dv[None], n_draw, 0)).to(dev),
            steps=DDIM_STEPS, guidance=1.0, seed=7 + col)
        Zs = to_yards(z.cpu().numpy(), COORDS)
        tr = reconstruct((120.0, 0.0), (ex, ey), float(dur[tech]),
                         height_class=height, technique=tech, n_points=60)
        traj = pd.DataFrame({'x': tr['x'], 'y': tr['y'], 'z': tr['z']})
        for r in range(n_draw):
            g = to_player_frame(ids, Zs[r:r + 1], F, M, [ti], 'q')
            draw(axes[r, col], g, traj, f'{name}\ndraw {r + 1}' if r == 0
                 else f'draw {r + 1}')
        # population response: every holdout roster, one draw per query
        zz = []
        for s in range(0, len(hold), 128):
            sl = hold[s:s + 128]
            zz.append(diff.ddim_sample(
                model, torch.tensor(F[sl]).to(dev), torch.tensor(M[sl]).to(dev),
                torch.full((len(sl),), 2, dtype=torch.long, device=dev),
                torch.tensor(np.repeat(dv[None], len(sl), 0)).to(dev),
                steps=100, guidance=1.0, seed=99 + s).cpu().numpy())
        P = to_player_frame(ids, to_yards(np.concatenate(zz), COORDS), F, M, hold, 'q')
        ax = axes[n_draw, col]
        for team, c in [(True, 'tab:red'), (False, 'tab:blue')]:
            s = P[(P.teammate == team) & ~P.actor & ~P.keeper]
            ax.hexbin(s.x, s.y, gridsize=(14, 30), extent=(90, 122, -2, 82),
                      cmap='Reds' if team else 'Blues', mincnt=3, alpha=0.55)
        draw(ax, P.iloc[:0], traj, f'all {len(hold)} holdout rosters\n'
             '(red = attackers, blue = defenders)')
    fig.suptitle('Delivery-conditioned generator: same roster, three requested '
                 'deliveries\n(label unconditioned; orange = requested ball, '
                 'X = arrival point)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{SP}/figures/delivery_conditioning{tag}.png', dpi=130)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    main(tag=ap.parse_args().tag)

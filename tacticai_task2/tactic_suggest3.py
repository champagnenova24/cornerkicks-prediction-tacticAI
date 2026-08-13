"""Joint recommendation: attacking shape AND delivery type, given the defence.

Extends the paper's Fig. 3 analog in two directions.

1. CONDITIONED ON THE DEFENCE. `ddim_inpaint` already pins chosen slots and
   re-denoises the rest, so freezing every defender plus the taker and sampling
   only the outfield attackers is a conditional draw from
   P(attackers | defenders, taker, shot=1). The denoiser is a permutation-
   equivariant transformer over all 22 slots, so attackers attend to the pinned
   defenders and the conditioning actually propagates. Best-of-N by the scorer,
   since a single guided draw is not reliably an improvement.

2. DELIVERY TYPE. Technique is not a free variable. Corner side crossed with
   the taker's foot determines the swing -- in 2,153 long corners there is not
   one counterexample:

       y=0  side + left foot  -> 93% outswinging, 0% inswinging
       y=0  side + right foot -> 92% inswinging,  0% outswinging
       y=80 side + left foot  -> 94% inswinging,  0% outswinging
       y=80 side + right foot -> 84% outswinging, 0% inswinging

   So "recommend a corner type" is really "recommend which foot takes it",
   which is a decision a coach can actually act on. After the tacticai
   reflection (kick mirrored to (120, 0)) this collapses to: effective right
   foot -> inswinging, effective left foot -> outswinging.

   Note `foot_right` in GDELIV is the RAW body part while the geometry is
   reflected, so unlike `tech_in`/`tech_out` (invariant -- an inswinger is an
   inswinger from either side) it is not reflection-invariant. Counterfactuals
   here set the effective foot and derive the raw flag from it.

WHAT THE DELIVERY RECOMMENDATION IS AND IS NOT
----------------------------------------------
The scorer is observational. Outswingers show a 29.0% shot rate against 18.8%
for inswingers, but teams choose the taker, and that choice travels with
personnel, game state and opponent. Reading the raw gap as "switch to an
outswinger and gain 10 points of shot probability" is exactly the confounded
inference the pipeline's positivity work exists to prevent. This script
therefore reports the raw contrast next to an IPW-adjusted one, and states the
shape gain and the delivery gain separately so they can be trusted differently:
the shape half is a within-corner intervention scored by a fixed model, the
delivery half is an observational contrast with partial adjustment.

Outputs: tactic_joint.csv, tactic_joint.png
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from diffusion_corners import SP, load_dataset, load_model, denormalize
from diffusion_eval import to_player_frame, real_player_frame
from tactic_suggest import ddim_inpaint, draw_panel
from train_eval import (build_features, CONFIGS, GCOLS_BASE, GVEL, GDELIV)

N_CAND = 6            # candidate attacking shapes per corner
W = 2.0
TRIM = (0.05, 0.95)
SCORE_COLS = GCOLS_BASE + GVEL + GDELIV
SETUP_COLS = GCOLS_BASE + GVEL

# In the reflected frame the swing follows from the effective foot.
TECH_OF_FOOT = {'Right Foot': 'Inswinging', 'Left Foot': 'Outswinging'}
OPPOSITE = {'Right Foot': 'Left Foot', 'Left Foot': 'Right Foot'}


def effective_foot(body_part, reflected):
    """Foot as seen in the mirrored frame (kick always at (120, 0))."""
    if body_part not in OPPOSITE:
        return None
    return OPPOSITE[body_part] if reflected else body_part


def raw_foot(eff_foot, reflected):
    return OPPOSITE[eff_foot] if reflected else eff_foot


def setup_globals(dpf):
    """Global setup features for a player frame (no delivery attributes)."""
    _, dg = build_features(None, dpf)
    return dg


def with_delivery(dg, tech, height, foot_right):
    """Attach delivery one-hots, taking explicit (possibly counterfactual)
    values rather than reading the observed ones off the parquet."""
    out = dg.copy()
    out['tech_in'] = float(tech == 'Inswinging')
    out['tech_out'] = float(tech == 'Outswinging')
    out['tech_straight'] = float(tech == 'Straight')
    out['tech_missing'] = float(tech not in
                                ('Inswinging', 'Outswinging', 'Straight'))
    out['h_high'] = (height == 'High Pass').astype(float) \
        if hasattr(height, 'astype') else float(height == 'High Pass')
    out['h_ground'] = float(height == 'Ground Pass')
    out['h_low'] = float(height == 'Low Pass')
    out['foot_right'] = float(foot_right)
    return out


def technique_effect(dg_real, meta, y):
    """Raw vs IPW-adjusted contrast: outswinging against inswinging.

    Treatment is the delivery the taker's foot produces. Adjustment uses the
    setup at the kick -- the geometry both teams can see. Personnel quality,
    game state and opponent are NOT observed here, so this closes part of the
    backdoor, not all of it.
    """
    m = meta.loc[dg_real.index]
    keep = m.technique.isin(['Inswinging', 'Outswinging'])
    X = dg_real.loc[keep, SETUP_COLS].to_numpy(float)
    t = (m.loc[keep, 'technique'] == 'Outswinging').to_numpy().astype(int)
    yy = y.loc[dg_real.index][keep].to_numpy(float)

    X = np.nan_to_num(X)
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    lr = LogisticRegression(max_iter=2000).fit(Xs, t)
    ps = lr.predict_proba(Xs)[:, 1]

    naive = yy[t == 1].mean() - yy[t == 0].mean()
    sel = (ps > TRIM[0]) & (ps < TRIM[1])
    tt, yt, pt = t[sel], yy[sel], ps[sel]
    w = np.where(tt == 1, 1 / pt, 1 / (1 - pt))
    ipw = ((w * yt * tt).sum() / (w * tt).sum()
           - (w * yt * (1 - tt)).sum() / (w * (1 - tt)).sum())
    return {
        'n': int(keep.sum()), 'n_out': int(t.sum()), 'n_in': int((1 - t).sum()),
        'ps_auc': roc_auc_score(t, ps), 'kept_after_trim': int(sel.sum()),
        'raw': naive, 'ipw': ipw,
    }


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f'{SP}/diffusion_split.csv')
    hold_idx = np.where(split.split.to_numpy() == 'holdout')[0]
    train_idx = np.where(split.split.to_numpy() == 'train')[0]
    model, diff = load_model(device)

    corners = pd.read_parquet(f'{SP}/corners.parquet').set_index('corner_id')
    deliv = pd.read_parquet(f'{SP}/corner_delivery.parquet').set_index('corner_id')
    meta = deliv.join(corners[['reflected']])
    meta['eff_foot'] = [effective_foot(b, r) for b, r
                        in zip(meta.body_part, meta.reflected)]
    shot_by = pd.Series(dict(zip(ids, y)))

    # ---------- scorer: delivery-aware, real training corners only ----------
    dp_tr = real_player_frame(ids, X, F, M, train_idx)
    dg_tr = setup_globals(dp_tr)
    md = meta.reindex(dg_tr.index)
    G_tr = dg_tr.copy()
    G_tr['tech_in'] = (md.technique == 'Inswinging').astype(float)
    G_tr['tech_out'] = (md.technique == 'Outswinging').astype(float)
    G_tr['tech_straight'] = (md.technique == 'Straight').astype(float)
    G_tr['tech_missing'] = (~md.technique.isin(
        ['Inswinging', 'Outswinging', 'Straight'])).astype(float)
    G_tr['h_high'] = (md.height == 'High Pass').astype(float)
    G_tr['h_ground'] = (md.height == 'Ground Pass').astype(float)
    G_tr['h_low'] = (md.height == 'Low Pass').astype(float)
    G_tr['foot_right'] = (md.body_part == 'Right Foot').astype(float)
    y_tr = shot_by[G_tr.index].to_numpy()
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=40, random_state=0).fit(
        G_tr[SCORE_COLS].to_numpy(float), y_tr)
    print(f'delivery-aware scorer trained on {len(G_tr):,} real corners')

    # ---------- is the delivery contrast trustworthy at all? ----------
    eff = technique_effect(dg_tr, meta, shot_by)
    print(f'\n=== technique contrast (outswinging - inswinging), training set ===')
    print(f'  n = {eff["n"]:,}  ({eff["n_out"]:,} out / {eff["n_in"]:,} in), '
          f'propensity AUC {eff["ps_auc"]:.3f}, '
          f'{eff["kept_after_trim"]:,} kept after trimming')
    print(f'  raw difference in shot rate   {eff["raw"]:+.4f}')
    print(f'  IPW adjusted on setup         {eff["ipw"]:+.4f}   '
          f'({(1 - eff["ipw"]/eff["raw"])*100:.0f}% of the raw gap was '
          f'setup confounding)')
    print('  Unadjusted: personnel, game state, opponent. Treat the delivery')
    print('  half of any recommendation below as suggestive, not causal.')

    # ---------- candidate attacking shapes, conditioned on the defence ------
    print(f'\ngenerating {N_CAND} attacking shapes per corner for '
          f'{len(hold_idx)} corners, defence pinned (w={W})', flush=True)
    dev = torch.device(device)
    cands = []
    for c in range(N_CAND):
        out = []
        for s in range(0, len(hold_idx), 128):
            sl = hold_idx[s:s + 128]
            x0 = torch.tensor(X[sl]).to(dev)
            f = torch.tensor(F[sl]).to(dev)
            m = torch.tensor(M[sl]).to(dev)
            team, actor = f[..., 0] > 0.5, f[..., 2] > 0.5
            move = team & ~actor & (m > 0.5)   # outfield attackers only
            lab = torch.ones(len(sl), dtype=torch.long, device=dev)
            z = ddim_inpaint(diff, model, x0, ~move, f, m, lab,
                             seed=7000 + 100 * c + s)
            out.append(z.cpu().numpy())
        cands.append(denormalize(np.concatenate(out)))
        print(f'  shape {c + 1}/{N_CAND} done', flush=True)

    # ---------- score every (shape, delivery) pair ----------
    dp_orig = real_player_frame(ids, X, F, M, hold_idx)
    dg_orig = setup_globals(dp_orig)
    dg_cand = [setup_globals(to_player_frame(ids, Z, F, M, hold_idx, f'c{c}'))
               for c, Z in enumerate(cands)]

    rows = []
    for k, i in enumerate(hold_idx):
        cid = ids[i]
        if cid not in meta.index or cid not in dg_orig.index:
            continue
        r = meta.loc[cid]
        ef = r.eff_foot
        if ef is None or r.technique not in ('Inswinging', 'Outswinging'):
            continue          # short corners / no swing: different decision
        obs_tech = TECH_OF_FOOT[ef]
        height = r.height

        def score(dg_row, tech, foot_eff):
            g = with_delivery(dg_row, tech, height,
                              raw_foot(foot_eff, r.reflected) == 'Right Foot')
            return float(clf.predict_proba(
                g[SCORE_COLS].to_numpy(float).reshape(1, -1))[0, 1])

        p_orig = score(dg_orig.loc[cid], obs_tech, ef)
        best = {'p': -1.0}
        for c, dgc in enumerate(dg_cand):
            key = f'{cid}__c{c}'
            if key not in dgc.index:
                continue
            for foot in ('Right Foot', 'Left Foot'):
                p = score(dgc.loc[key], TECH_OF_FOOT[foot], foot)
                if p > best['p']:
                    best = {'p': p, 'cand': c, 'foot': foot}
        if best['p'] < 0:
            continue
        # shape-only gain holds the delivery at what was actually used.
        # Keep the MEAN over candidates too: the max is chosen by the same
        # scorer that then reports it, so max-minus-mean is the part of the
        # headline gain that is selection on a noisy score rather than the
        # guidance genuinely moving setups.
        shape_scores = [score(dgc.loc[f'{cid}__c{c}'], obs_tech, ef)
                        for c, dgc in enumerate(dg_cand)
                        if f'{cid}__c{c}' in dgc.index]
        p_shape = max(shape_scores)
        p_shape_mean = float(np.mean(shape_scores))
        rows.append(dict(
            corner_id=cid, shot_real=int(y[i]), reflected=bool(r.reflected),
            obs_foot_eff=ef, obs_technique=obs_tech,
            rec_foot_eff=best['foot'], rec_technique=TECH_OF_FOOT[best['foot']],
            rec_cand=best['cand'], p_original=p_orig,
            p_shape_only=p_shape, p_shape_mean=p_shape_mean,
            p_joint=best['p'], switched=int(best['foot'] != ef)))

    df = pd.DataFrame(rows)
    df['gain_shape'] = df.p_shape_only - df.p_original
    df['gain_shape_mean'] = df.p_shape_mean - df.p_original
    df['gain_delivery'] = df.p_joint - df.p_shape_only
    df['gain_total'] = df.p_joint - df.p_original
    df.to_csv(f'{SP}/tactic_joint.csv', index=False)

    print(f'\n=== joint recommendation, {len(df)} swung holdout corners ===')
    print(f'  mean P(shot)  original {df.p_original.mean():.3f}'
          f'  -> best shape {df.p_shape_only.mean():.3f}'
          f'  -> + best delivery {df.p_joint.mean():.3f}')
    print(f'  gain from shape    {df.gain_shape.mean():+.3f}  '
          f'(improved on {(df.gain_shape > 0).mean():.0%} of corners)')
    print(f'  gain from delivery {df.gain_delivery.mean():+.3f}  '
          f'(recommends switching foot on {df.switched.mean():.0%})')

    print(f'\n  how much of the shape gain is real vs selection:')
    print(f'    mean over the {N_CAND} candidates  {df.gain_shape_mean.mean():+.3f}'
          f'   <- guidance actually moving setups')
    print(f'    max  over the {N_CAND} candidates  {df.gain_shape.mean():+.3f}'
          f'   <- what best-of-{N_CAND} reports')
    frac = (1 - df.gain_shape_mean.mean() / df.gain_shape.mean()) * 100
    print(f'    => {frac:.0f}% of the headline shape gain is selection on a '
          f'noisy scorer')

    print('\n  recommended technique by observed technique:')
    ct = pd.crosstab(df.obs_technique, df.rec_technique)
    print(ct.to_string())
    share = (df.rec_technique == 'Outswinging').mean()
    print(f'    recommends outswinging on {share:.0%} of corners -- a policy '
          f'this close to constant is')
    print(f'    reproducing the marginal shot-rate gap, not reading the '
          f'defence in front of it.')

    # ---------- figure ----------
    pick = df.sort_values('gain_total', ascending=False).iloc[0]
    cid = pick.corner_id
    o = dp_orig[dp_orig.corner_id == cid]
    c = to_player_frame(ids, cands[int(pick.rec_cand)], F, M, hold_idx,
                        f'c{int(pick.rec_cand)}')
    a = c[c.corner_id == f'{cid}__c{int(pick.rec_cand)}']
    po = o[o.teammate & ~o.actor]
    pa = a[a.teammate & ~a.actor]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    draw_panel(axes[0], o, f'original: {pick.obs_technique.lower()}, '
                           f'P={pick.p_original:.2f}')
    draw_panel(axes[1], a,
               f'recommended: {pick.rec_technique.lower()} '
               f'({pick.rec_foot_eff.split()[0].lower()} foot), '
               f'P={pick.p_joint:.2f}',
               moved_pairs=list(zip(po.x, po.y, pa.x, pa.y)))
    fig.suptitle('Joint recommendation: attacking shape (defence pinned) + '
                 'delivery type\nblue = defence, held fixed; gray = original '
                 'attacker position -> recommended', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{SP}/tactic_joint.png', dpi=150)
    print('\nwrote tactic_joint.csv, tactic_joint.png')


if __name__ == '__main__':
    main()

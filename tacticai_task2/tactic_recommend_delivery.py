"""The corner recommendation rebuilt on the delivery-aware generator.

`tactic_recommend.py` recommends a delivery target (4-9 yd, central through
far post) and an attacking shape -- but it draws the shape with the
delivery-BLIND generator, guided toward shot = 1, and then pastes the
recommended ball on top. The shape never knew which ball it was for. That is
the "column in a one-hot" problem the README flagged for `tactic_suggest3.py`,
one level up.

`diffusion_delivery.py` can be told the ball. So here, with every defender and
the taker pinned to their real positions (inpainting), the attackers are
re-drawn FOR a specific delivery:

  as played    the real freeze frame
  old          tactic_recommend.py's method: delivery-blind model, shot = 1,
               guidance w = 2
  observed     delivery model, told the ball that was actually played
  recommended  delivery model, told the recommended ball: same taker and swing,
               arrival at the empirical centre of the target region, High Pass
  defence      the reverse query: attackers pinned, DEFENDERS re-drawn for the
               recommended ball -- how defences have set up before such balls

K independent draws per arm, averaged: `tactic_recommend.py` showed that
choosing among draws adds nothing once selection bias is removed, so nothing
is selected here.

What is measured, and why not xG. Per-corner xG is not predictable from the
setup (out-of-sample R^2 < 0, see the README), so an xG scorer would rank
shapes by noise. The question this generator can answer is structural: does
the shape serve the ball? Per corner and arm, at the kick (no velocities):

  att_near_target   an attacker within 3 yd of the recommended arrival point
  att_minus_def     attackers minus defenders within 5 yd of it -- local
                    numerical advantage where the ball is going
  att_in_region     attackers inside the recommended target region

plus, for continuity with tactic_recommend.py, P(shot) from the cross-fitted
evaluator -- in the scorer's units, with its known caveats.

The leakage boundary still applies: a delivery-conditioned shape is "how
attackers have set up before balls like this", learned from setups that were
followed by that ball. It is a coaching sketch, not a causal effect.

Outputs: results/tactic_recommendation_delivery.csv,
         figures/tactic_recommendation_delivery.png
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

import outdirs  # noqa: F401 -- creates results/ figures/ models/
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SP), 'pipeline'))

from ball_trajectory import reconstruct                      # noqa: E402
from diffusion_corners import (pick_device, load_dataset, MAXP, CH,  # noqa: E402
                               T_STEPS, load_model as load_blind)
import diffusion_delivery as dd                              # noqa: E402
from diffusion_eval import to_player_frame, real_player_frame, globals_for  # noqa: E402
from tactic_suggest import ddim_inpaint as ddim_inpaint_blind  # noqa: E402
from tactic_recommend import (target_point_and_flight, TARGET_DEPTH,  # noqa: E402
                              TARGET_LATERAL, MIN_CHORD)
from train_eval import CONFIGS                               # noqa: E402

K = 3                 # draws per arm, averaged -- never selected among
STEPS = 200
NEAR_YD = 3.0
LOCAL_YD = 5.0
ARMS = ['as played', 'old', 'observed', 'recommended']


@torch.no_grad()
def inpaint(diff, model, x0, known, flags, mask, deliv, seed, steps=STEPS):
    """RePaint-style single pass for the delivery model: `known` slots are
    re-noised from their real values at every step, the rest are denoised
    conditioned on `deliv` (label unconditioned, w = 1)."""
    dev = diff.device
    g = torch.Generator(device='cpu').manual_seed(seed)
    B = flags.shape[0]
    x = torch.randn(B, MAXP, CH, generator=g).to(dev)
    null = torch.full((B,), 2, dtype=torch.long, device=dev)
    has = torch.ones(B, device=dev)
    ts = torch.linspace(T_STEPS - 1, 0, steps).long().to(dev)
    kn = known[..., None]
    for i in range(steps):
        ab = diff.alpha_bar[ts[i]]
        noise = torch.randn(B, MAXP, CH, generator=g).to(dev)
        x = torch.where(kn, ab.sqrt() * x0 + (1 - ab).sqrt() * noise, x)
        eps = model(x, flags, mask, ts[i].repeat(B), null, deliv, has)
        x0h = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-2.5, 2.5)
        if i < steps - 1:
            ab_p = diff.alpha_bar[ts[i + 1]]
            x = ab_p.sqrt() * x0h + (1 - ab_p).sqrt() * eps
        else:
            x = torch.where(kn, x0, x0h)
    return x


def structure(players, tx, ty, T):
    """Per-corner structural metrics around the recommended arrival point:
    from positions at the kick, and at arrival (position + velocity x the
    recommended flight time T, as in synthetic_trajectories.py -- velocities
    are flight-window averages, so this is where a player is when the ball
    lands)."""
    rows = []
    for cid, g in players.groupby('corner_id', sort=False):
        att = g[g.teammate & ~g.actor]
        da_arr = np.hypot(att.x + att.vx * T - tx, att.y + att.vy * T - ty)
        dfd = g[~g.teammate & ~g.keeper]
        da = np.hypot(att.x - tx, att.y - ty)
        dd_ = np.hypot(dfd.x - tx, dfd.y - ty)
        depth = 120.0 - att.x
        rows.append(dict(
            corner_id=cid.split('__')[0],
            att_near_target=float((da < NEAR_YD).any()),
            att_near_target_arrival=float((da_arr < NEAR_YD).any()),
            att_dist_target=float(da.min()) if len(da) else np.nan,
            att_minus_def=float((da < LOCAL_YD).sum() - (dd_ < LOCAL_YD).sum()),
            def_near_target=float((dd_ < NEAR_YD).any()),
            att_in_region=float(((depth >= TARGET_DEPTH[0])
                                 & (depth < TARGET_DEPTH[1])
                                 & (att.y >= TARGET_LATERAL[0])
                                 & (att.y < TARGET_LATERAL[1])).sum())))
    return pd.DataFrame(rows)


def main(device=None):
    device = device or pick_device()
    dev = torch.device(device)
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f'{SP}/results/diffusion_split.csv')
    hold = np.where(split.split.to_numpy() == 'holdout')[0]
    train = np.where(split.split.to_numpy() == 'train')[0]

    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet')
    dl['chord'] = np.hypot(dl.end_x - dl.start_x, dl.end_y - dl.start_y)
    tx, ty, tdur, n_zone = target_point_and_flight(dl)
    dl = dl.set_index('corner_id')
    hold = np.array([i for i in hold if dl.loc[ids[i], 'chord'] >= MIN_CHORD])
    print(f'{len(hold)} long holdout corners; recommended arrival ({tx:.1f}, '
          f'{ty:.1f}), from {n_zone} real deliveries into the target region')

    model, diff = dd.load_model(device)
    assert getattr(model, 'coords', 'linear') == 'linear', \
        'the recommender uses the linear-coordinate checkpoint'
    blind, bdiff = load_blind(device)
    D_obs = dd.load_deliveries([ids[i] for i in hold])
    D_rec = np.stack([dd.encode_delivery(
        tx, ty, None if pd.isna(dl.loc[ids[i], 'technique'])
        else dl.loc[ids[i], 'technique'], 'High Pass') for i in hold])

    frames = {'as played': real_player_frame(ids, X, F, M, hold)}
    draws = {a: [] for a in ['old', 'observed', 'recommended', 'defence']}
    for k in range(K):
        out = {a: [] for a in draws}
        for s in range(0, len(hold), 128):
            sl = hold[s:s + 128]
            x0 = torch.tensor(X[sl]).to(dev)
            f = torch.tensor(F[sl]).to(dev)
            m = torch.tensor(M[sl]).to(dev)
            team, actor = f[..., 0] > 0.5, f[..., 2] > 0.5
            move_att = team & ~actor & (m > 0.5)
            move_def = ~team & (f[..., 1] < 0.5) & (m > 0.5)   # outfield defenders
            seed = 7000 + 100 * k + s
            lab = torch.ones(len(sl), dtype=torch.long, device=dev)
            out['old'].append(ddim_inpaint_blind(
                bdiff, blind, x0, ~move_att, f, m, lab, seed=seed).cpu().numpy())
            for arm, D, known in [('observed', D_obs, ~move_att),
                                  ('recommended', D_rec, ~move_att),
                                  ('defence', D_rec, ~move_def)]:
                dv = torch.tensor(D[s:s + 128]).to(dev)
                out[arm].append(inpaint(diff, model, x0, known, f, m, dv,
                                        seed=seed).cpu().numpy())
        for a in draws:
            Z = dd.to_yards(np.concatenate(out[a]), 'linear')
            draws[a].append(to_player_frame(ids, Z, F, M, hold, f'{a}{k}'))
        print(f'  draw {k + 1}/{K} done', flush=True)

    # ---- structural metrics, averaged over draws --------------------------
    S = {'as played': structure(frames['as played'], tx, ty, tdur)}
    for a in draws:
        S[a] = pd.concat([structure(d, tx, ty, tdur) for d in draws[a]]) \
                 .groupby('corner_id', sort=False).mean().reset_index()

    # ---- cross-fitted P(shot), as in tactic_recommend.py -------------------
    gcols = CONFIGS['vel'][1]
    shot_by = dict(zip(ids, y))
    perm = np.random.default_rng(0).permutation(train)
    G_ev = globals_for(real_player_frame(ids, X, F, M, perm[len(perm) // 2:]), gcols)
    ev = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=40,
        random_state=1).fit(G_ev.to_numpy(float),
                            np.array([shot_by[c] for c in G_ev.index]))

    def p_ev(frame):
        G = globals_for(frame, gcols)
        return pd.Series(ev.predict_proba(G.to_numpy(float))[:, 1],
                         index=[c.split('__')[0] for c in G.index])

    P = {'as played': p_ev(frames['as played'])}
    for a in draws:
        P[a] = pd.concat([p_ev(d) for d in draws[a]]).groupby(level=0).mean()

    rows = []
    for a in ['as played', 'old', 'observed', 'recommended', 'defence']:
        t = S[a].set_index('corner_id')
        t = t.join(P[a].rename('p_shot_ev'))
        t['arm'] = a
        rows.append(t.reset_index())
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(f'{SP}/results/tactic_recommendation_delivery.csv', index=False)

    summ = res.groupby('arm', sort=False)[
        ['att_near_target', 'att_near_target_arrival', 'att_dist_target',
         'att_minus_def', 'att_in_region', 'def_near_target', 'p_shot_ev']].mean()
    summ['att_near_target'] *= 100
    summ['att_near_target_arrival'] *= 100
    summ['def_near_target'] *= 100
    print(f'\n=== around the recommended arrival point, at the kick '
          f'({len(hold)} corners, {K} draws per arm) ===')
    print(summ.round(3).to_string())

    # paired contrast with a match-cluster bootstrap: recommended vs old
    mid = dl.match_id
    piv = res.pivot_table(index='corner_id', columns='arm',
                          values=['att_near_target', 'att_near_target_arrival',
                                  'att_minus_def'])
    piv['match'] = mid.reindex(piv.index).to_numpy()
    rng = np.random.default_rng(0)
    # Two contrasts. recommended - old is the whole upgrade; recommended -
    # observed isolates what being told the RECOMMENDED ball (rather than the
    # one actually played) adds, on corners whose real ball was far from it.
    far = dl.reindex(piv.index)
    far = np.hypot(far.end_x - tx, far.end_y - ty) > 8.0
    for a, b, sub, lab in [('recommended', 'old', slice(None), 'all corners'),
                           ('recommended', 'observed', far,
                            'real ball > 8 yd from the target')]:
        pv = piv[sub]
        grp = {m: g for m, g in pv.groupby('match')}
        um_ = np.array(list(grp))
        print(f'\n  {a} minus {b}, {lab} (n={len(pv)}; paired, 1000 '
              f'match-cluster bootstraps):')
        for met in ['att_near_target', 'att_near_target_arrival', 'att_minus_def']:
            diffs = pv[(met, a)] - pv[(met, b)]
            bs = []
            for _ in range(1000):
                pick = rng.choice(um_, len(um_))
                bs.append(np.concatenate([
                    (grp[m][(met, a)] - grp[m][(met, b)]).to_numpy()
                    for m in pick]).mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            scale = 1 if met == 'att_minus_def' else 100
            print(f'    {met:<24} {diffs.mean() * scale:+.3f} '
                  f'[{lo * scale:+.3f}, {hi * scale:+.3f}]'
                  f'{"  *" if not lo < 0 < hi else ""}')

    figure(ids, F, M, hold, dl, frames, draws, S, tx, ty, tdur)
    print('\nwrote results/tactic_recommendation_delivery.csv, '
          'figures/tactic_recommendation_delivery.png')


def figure(ids, F, M, hold, dl, frames, draws, S, tx, ty, tdur):
    """Four corners: as played / old recommender / new recommender /
    defence re-drawn for the recommended ball."""
    from synthetic_trajectories import draw
    far = S['as played'].set_index('corner_id').att_dist_target
    pick = []
    for tech in ('Inswinging', 'Outswinging'):
        cand = [ids[i] for i in hold if dl.loc[ids[i], 'technique'] == tech]
        cand = sorted(cand, key=lambda c: -far.get(c, 0))   # target was empty
        pick += cand[:2]
    titles = ['as played', 'old recommender\n(ball-blind shape)',
              'new: attackers re-drawn\nFOR the recommended ball',
              'defence re-drawn for\nthe recommended ball']
    fig, axes = plt.subplots(len(pick), 4, figsize=(14, 4.6 * len(pick)))
    for r, cid in enumerate(pick):
        d = dl.loc[cid]
        tech = d.technique if d.technique in ('Inswinging', 'Outswinging',
                                              'Straight') else None
        obs = reconstruct((d.start_x, d.start_y), (d.end_x, d.end_y), d.duration,
                          height_class=d.height_class, technique=tech,
                          end_height=d.end_height)
        rec = reconstruct((d.start_x, d.start_y), (tx, ty), tdur,
                          height_class='High Pass', technique=tech,
                          end_height=d.end_height)
        to_df = lambda t: pd.DataFrame({'x': t['x'], 'y': t['y'], 'z': t['z']})
        panels = [(frames['as played'], '', obs)] + [
            (draws[a][0], f'__{a}0', rec) for a in ['old', 'recommended', 'defence']]
        for c, (fr, suf, traj) in enumerate(panels):
            g = fr[fr.corner_id == f'{cid}{suf}']
            ax = axes[r, c]
            ax.add_patch(plt.Rectangle(
                (120 - TARGET_DEPTH[1], TARGET_LATERAL[0]),
                TARGET_DEPTH[1] - TARGET_DEPTH[0],
                TARGET_LATERAL[1] - TARGET_LATERAL[0],
                facecolor='gold', alpha=0.25, edgecolor='goldenrod', zorder=1))
            draw(ax, g, to_df(traj),
                 f'{titles[c]}\n{str(d.technique).lower()}' if r == 0 else
                 str(d.technique).lower())
    fig.suptitle('Corner recommendation on the delivery-aware generator: defence '
                 'pinned, attackers re-drawn for the recommended ball\n'
                 '(gold = recommended target region; orange = ball; column 1 '
                 'shows the ball actually played)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{SP}/figures/tactic_recommendation_delivery.png', dpi=120)


if __name__ == '__main__':
    main()

"""Corner recommendation: attacking shape, delivery type, and the ball path.

Supersedes the delivery half of `tactic_suggest3.py`, which maximised P(shot).
That was faithful to the paper's Task 2 target but wrong as tactical advice:
adjusted for the defensive setup and clustered over matches, outswinging buys
+0.105 shot attempts per corner [+0.052, +0.154] and no expected goals at all
(+0.003 xG, CI [-0.009, +0.016]) and no goals (-0.003, CI [-0.024, +0.021]).
Inswinger attempts are worth ~0.25 xG each against ~0.17 for outswingers: the
extra outswinger shots are cheaper shots, not more goals. Optimising P(shot)
therefore recommended switching the taker's foot on 67% of corners in exchange
for nothing that shows up on the scoreboard.

So the delivery recommendation here is built on xG and on the zone table, not
on the shot classifier:

  WHERE  central, 4-9 yd off the goal line (the six-yard line across the
         middle of the goal). Best-supported zone in the table (n=565) and the
         highest-value lateral band in every defensive state tested.
  WHICH  no evidence to switch the taker. Technique moves attempts, not value.
  SHAPE  attackers re-drawn by the diffusion model with every defender pinned.

The ball path for both the observed and the recommended delivery is
reconstructed with the pipeline's `ball_trajectory.reconstruct`, so the output
carries a flight time and an apex, not just an end point. The swing magnitude
in that reconstruction is a prior, not a measurement -- see the pipeline README.

Outputs: tactic_recommendation.csv, tactic_recommendation.png
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

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SP), 'pipeline'))

from ball_trajectory import reconstruct                      # noqa: E402
from diffusion_corners import load_dataset, load_model, denormalize  # noqa: E402
from diffusion_eval import to_player_frame, real_player_frame, globals_for  # noqa: E402
from tactic_suggest import ddim_inpaint                      # noqa: E402
from train_eval import CONFIGS                               # noqa: E402

N_CAND = 6
W = 2.0
MIN_CHORD = 15.0            # below this it is a short corner: another decision

# Recommended target: the 4-9 yd band, central through far post.
#
# Revised on the FULL StatsBomb set (41,178 corners, `delivery_target_full.py`).
# On the 2,473-corner 360 subset central looked dominant (0.0561 vs 0.0446 far
# post) and this was (36, 44). With 4.3x tighter CIs that separation does not
# survive: far post 0.0480 [0.0453, 0.0509] against central 0.0458 [0.0442,
# 0.0473], and the paired contrast crosses zero (-0.0022 [-0.0054, +0.0009]).
# The two are indistinguishable, so the target is the union of them rather than
# a false-precision point inside one. What DID survive, and hardened: the near
# post is worse than central by +0.0101 [+0.0078, +0.0123] and takes 39% of all
# long corners, and 0-4 yd (right on top of the goal) is worse than 4-9 yd.
TARGET_DEPTH = (4.0, 9.0)
TARGET_LATERAL = (36.0, 52.0)


def zone_of_delivery(end_x, end_y):
    depth = 120.0 - end_x
    if depth < 4:
        d = '0-4yd'
    elif depth < 9:
        d = '4-9yd'
    else:
        d = '9yd+'
    if end_y < 36:
        l = 'near post'
    elif end_y < 44:
        l = 'central'
    elif end_y < 52:
        l = 'far post'
    else:
        l = 'beyond far post'
    return f'{d} x {l}'


def target_point_and_flight(dl):
    """Empirical centre of the recommended zone, and the flight it takes.

    Taken from real deliveries that actually landed there rather than the
    geometric centre of the box, so the recommendation is a delivery someone
    has demonstrably hit.
    """
    depth = 120.0 - dl.end_x
    inzone = ((depth >= TARGET_DEPTH[0]) & (depth < TARGET_DEPTH[1])
              & (dl.end_y >= TARGET_LATERAL[0]) & (dl.end_y < TARGET_LATERAL[1])
              & (dl.chord >= MIN_CHORD))
    z = dl[inzone]
    return (float(z.end_x.median()), float(z.end_y.median()),
            float(z.duration.median()), len(z))


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ids, X, F, M, y = load_dataset()
    split = pd.read_csv(f'{SP}/diffusion_split.csv')
    hold_idx = np.where(split.split.to_numpy() == 'holdout')[0]
    train_idx = np.where(split.split.to_numpy() == 'train')[0]
    model, diff = load_model(device)

    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet')
    dl['chord'] = np.hypot(dl.end_x - dl.start_x, dl.end_y - dl.start_y)
    dl = dl.set_index('corner_id')
    dfn = pd.read_parquet(f'{SP}/defense_features.parquet').set_index('corner_id')

    tx, ty, tdur, n_zone = target_point_and_flight(dl.reset_index())
    print(f'recommended target: ({tx:.1f}, {ty:.1f}), median flight '
          f'{tdur:.2f}s, from {n_zone:,} real deliveries into that zone')

    # ---- two INDEPENDENT shape scorers, trained on disjoint corners --------
    # Selecting the best of N candidates with the same model that then reports
    # the gain is selection on that model's own noise: the argmax of N noisy
    # scores is biased upward even if every candidate is identical in truth.
    # Cross-fitting removes it -- `sel` picks the candidate, `ev` (which never
    # saw sel's corners) measures it, so ev's noise is independent of the
    # choice. The gap between the two readings IS the bias.
    gcols = CONFIGS['vel'][1]
    shot_by = dict(zip(ids, y))
    rng = np.random.default_rng(0)
    perm = rng.permutation(train_idx)
    half = len(perm) // 2

    def fit_scorer(idx_subset, seed):
        G = globals_for(real_player_frame(ids, X, F, M, idx_subset), gcols)
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=40, random_state=seed).fit(
            G.to_numpy(float), np.array([shot_by[c] for c in G.index]))

    clf_sel = fit_scorer(perm[:half], 0)
    clf_ev = fit_scorer(perm[half:], 1)
    print(f'selector trained on {half} corners, evaluator on '
          f'{len(perm) - half} disjoint corners')

    # ---- candidate attacking shapes, every defender pinned ----
    print(f'\ngenerating {N_CAND} shapes per corner for {len(hold_idx)} '
          f'corners, defence pinned (w={W})', flush=True)
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
            move = team & ~actor & (m > 0.5)
            lab = torch.ones(len(sl), dtype=torch.long, device=dev)
            out.append(ddim_inpaint(diff, model, x0, ~move, f, m, lab,
                                    seed=7000 + 100 * c + s).cpu().numpy())
        cands.append(denormalize(np.concatenate(out)))
        print(f'  shape {c + 1}/{N_CAND}', flush=True)

    dp_orig = real_player_frame(ids, X, F, M, hold_idx)
    G_orig = globals_for(dp_orig, gcols)
    dp_cand = [to_player_frame(ids, Z, F, M, hold_idx, f'c{c}')
               for c, Z in enumerate(cands)]
    G_cand = [globals_for(d, gcols) for d in dp_cand]

    rows = []
    cand_rng = np.random.default_rng(12345)   # picks the no-selection draw
    for i in hold_idx:
        cid = ids[i]
        if cid not in dl.index or cid not in G_orig.index:
            continue
        r = dl.loc[cid]
        if r.chord < MIN_CHORD:
            continue                     # short corner: out of scope
        tech = r.technique if r.technique in ('Inswinging', 'Outswinging',
                                              'Straight') else None
        obs = reconstruct((r.start_x, r.start_y), (r.end_x, r.end_y),
                          r.duration, height_class=r.height_class,
                          technique=tech, end_height=r.end_height)
        rec = reconstruct((r.start_x, r.start_y), (tx, ty), tdur,
                          height_class='High Pass', technique=tech,
                          end_height=r.end_height)
        avail = [c for c, g in enumerate(G_cand) if f'{cid}__c{c}' in g.index]
        if not avail:
            continue
        s_sel = np.array([float(clf_sel.predict_proba(
            G_cand[c].loc[[f'{cid}__c{c}']].to_numpy(float))[0, 1])
            for c in avail])
        s_ev = np.array([float(clf_ev.predict_proba(
            G_cand[c].loc[[f'{cid}__c{c}']].to_numpy(float))[0, 1])
            for c in avail])
        pick = int(np.argmax(s_sel))            # chosen WITHOUT seeing s_ev
        rand = int(cand_rng.integers(len(avail)))   # no selection at all
        d = dfn.loc[cid] if cid in dfn.index else None
        rows.append(dict(
            corner_id=cid, shot_real=int(y[i]),
            technique=r.technique, height=r.height_class,
            obs_end_x=r.end_x, obs_end_y=r.end_y,
            obs_zone=zone_of_delivery(r.end_x, r.end_y),
            obs_apex=obs['apex_z'], obs_duration=obs['duration'],
            rec_end_x=tx, rec_end_y=ty, rec_zone=zone_of_delivery(tx, ty),
            rec_apex=rec['apex_z'], rec_duration=rec['duration'],
            rec_technique='pick the taker on delivery quality, not swing',
            already_on_target=int(zone_of_delivery(r.end_x, r.end_y)
                                  == zone_of_delivery(tx, ty)),
            # measured by the evaluator, which had no hand in choosing
            p_ev_original=float(clf_ev.predict_proba(
                G_orig.loc[[cid]].to_numpy(float))[0, 1]),
            p_ev_selected=float(s_ev[pick]),
            p_ev_random=float(s_ev[rand]),
            p_ev_mean=float(s_ev.mean()),
            # the same quantities read off the selector: the biased view
            p_sel_original=float(clf_sel.predict_proba(
                G_orig.loc[[cid]].to_numpy(float))[0, 1]),
            p_sel_selected=float(s_sel[pick]),
            best_cand=avail[pick],
            def_in6=(float(d.def_in6) if d is not None else np.nan),
            mark_dist=(float(d.mark_dist) if d is not None else np.nan)))

    df = pd.DataFrame(rows)
    # honest: candidate chosen by sel, gain read by ev
    df['gain_honest'] = df.p_ev_selected - df.p_ev_original
    # no selection at all: one arbitrary draw
    df['gain_no_selection'] = df.p_ev_random - df.p_ev_original
    df['gain_mean_draw'] = df.p_ev_mean - df.p_ev_original
    # the number the old code reported: select and measure with one model
    df['gain_biased'] = df.p_sel_selected - df.p_sel_original
    df.to_csv(f'{SP}/tactic_recommendation.csv', index=False)

    print(f'\n=== recommendation over {len(df)} long holdout corners ===')
    print(f'  delivery target : ({tx:.1f}, {ty:.1f})  '
          f'= {zone_of_delivery(tx, ty)}')
    print(f'  reconstructed flight: {tdur:.2f}s, apex '
          f'{df.rec_apex.mean():.2f} yd')
    print(f'  corners already delivered into that zone: '
          f'{df.already_on_target.mean():.0%}')

    print(f'\n  --- shape gain, selection bias removed by cross-fitting ---')
    print(f'    one arbitrary draw (no selection)      '
          f'{df.gain_no_selection.mean():+.4f}')
    print(f'    mean over all {N_CAND} draws              '
          f'{df.gain_mean_draw.mean():+.4f}')
    print(f'    best-of-{N_CAND}, measured independently  '
          f'{df.gain_honest.mean():+.4f}   <- the honest number')
    print(f'    best-of-{N_CAND}, measured by the selector '
          f'{df.gain_biased.mean():+.4f}   <- what one model reports')
    infl = df.gain_biased.mean() - df.gain_honest.mean()
    print(f'    inflation from selecting and scoring with one model '
          f'{infl:+.4f}')
    sel_gain = df.gain_honest.mean() - df.gain_mean_draw.mean()
    print(f'    value added by CHOOSING among draws  {sel_gain:+.4f}  '
          f'({"real" if sel_gain > 0.005 else "negligible"})')
    print(f'    improved on {(df.gain_honest > 0).mean():.0%} of corners')

    print('\n  recommended technique (recommendation keeps the taker):')
    tv = df.technique.value_counts(normalize=True) * 100
    print(tv.round(1).to_string())
    print('    -> the distribution is the OBSERVED one. The recommendation is')
    print('       not "always inswing"; it is "no xG case for switching".')

    print('\n  observed delivery zone (where corners actually go):')
    vc = df.obs_zone.value_counts(normalize=True).head(6) * 100
    print(vc.round(1).to_string())

    # ---- figure: several examples, 2 inswinging and 2 outswinging ----------
    # Showing both techniques is the point: the recommendation preserves the
    # delivery it was given, so the panels must not all be one swing.
    def panel(ax, dpf, traj, title, moved=None, show_target=False):
        # The recommendation is a REGION, not the point the ball is drawn to.
        # Central and far post are statistically tied on the full set, and the
        # empirical centroid of their union sits central only because central
        # deliveries outnumber far-post ones ~4:1 there. Drawing just the
        # centroid would imply a precision the data does not support.
        if show_target:
            ax.add_patch(plt.Rectangle(
                (120 - TARGET_DEPTH[1], TARGET_LATERAL[0]),
                TARGET_DEPTH[1] - TARGET_DEPTH[0],
                TARGET_LATERAL[1] - TARGET_LATERAL[0],
                facecolor='gold', alpha=0.28, edgecolor='goldenrod',
                lw=1.2, zorder=1))
        ax.add_patch(plt.Rectangle((90, 0), 30, 80, fill=False, lw=1, ec='gray'))
        ax.plot([102, 102], [18, 62], c='gray', lw=1)
        ax.plot([102, 120], [18, 18], c='gray', lw=1)
        ax.plot([102, 120], [62, 62], c='gray', lw=1)
        ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, lw=1, ec='gray'))
        ax.plot([120, 120], [36, 44], c='black', lw=3)
        if moved:
            for xo, yo, xn, yn in moved:
                ax.plot([xo, xn], [yo, yn], c='dimgray', lw=0.9, alpha=.8, zorder=2)
                ax.scatter(xo, yo, facecolors='none', edgecolors='dimgray',
                           s=26, lw=.8, zorder=2)
        z = traj['z']
        ax.plot(traj['x'], traj['y'], c='darkorange', lw=1.3, alpha=.55, zorder=3)
        ax.scatter(traj['x'], traj['y'], s=2 + 10 * z / max(z.max(), 1e-6),
                   c='darkorange', alpha=.85, zorder=3)
        ax.scatter([traj['x'][-1]], [traj['y'][-1]], marker='X', s=110,
                   c='darkorange', edgecolors='black', linewidths=.7, zorder=6)
        for p in dpf.itertuples():
            col = 'tab:red' if p.teammate else 'tab:blue'
            mk = '*' if p.actor else ('s' if p.keeper else 'o')
            ax.scatter(p.x, p.y, c=col, marker=mk, s=75 if p.actor else 40,
                       zorder=5, edgecolors='black', linewidths=.4)
        ax.set_xlim(88, 122)
        ax.set_ylim(-2, 82)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9)

    picks = []
    for tech_name in ('Inswinging', 'Outswinging'):
        sub = df[(df.technique == tech_name) & (df.gain_honest > 0)] \
            .sort_values('gain_honest', ascending=False)
        if len(sub) >= 2:
            picks += [sub.iloc[0], sub.iloc[len(sub) // 2]]

    fig, axes = plt.subplots(len(picks), 2, figsize=(9.5, 4.9 * len(picks)))
    for row, pk in enumerate(picks):
        cid = pk.corner_id
        r = dl.loc[cid]
        tech = r.technique if r.technique in (
            'Inswinging', 'Outswinging', 'Straight') else None
        obs = reconstruct((r.start_x, r.start_y), (r.end_x, r.end_y),
                          r.duration, height_class=r.height_class,
                          technique=tech, end_height=r.end_height)
        rec = reconstruct((r.start_x, r.start_y), (tx, ty), tdur,
                          height_class='High Pass', technique=tech,
                          end_height=r.end_height)
        o = dp_orig[dp_orig.corner_id == cid]
        a = dp_cand[int(pk.best_cand)]
        a = a[a.corner_id == f'{cid}__c{int(pk.best_cand)}']
        po, pa = o[o.teammate & ~o.actor], a[a.teammate & ~a.actor]
        panel(axes[row, 0], o, obs,
              f'as played — {str(pk.technique).lower()} to {pk.obs_zone}\n'
              f'apex {obs["apex_z"]:.1f} yd, flight {obs["duration"]:.2f}s, '
              f'P={pk.p_ev_original:.2f}')
        panel(axes[row, 1], a, rec,
              f'recommended — SAME {str(pk.technique).lower()} taker, '
              f'ball to {pk.rec_zone}\napex {rec["apex_z"]:.1f} yd, flight '
              f'{rec["duration"]:.2f}s, P={pk.p_ev_selected:.2f}',
              moved=list(zip(po.x, po.y, pa.x, pa.y)), show_target=True)
    fig.suptitle(
        'Corner recommendations: attacking shape (defence pinned) + delivery '
        'target\nrows 1-2 inswinging, rows 3-4 outswinging — the technique is '
        'KEPT, only the target and the shape move\n'
        'gold band = recommended target REGION (4-9 yd, central through far post: '
        'the two are statistically tied);\norange = ball path (dot size = height), '
        'X = arrival; gray = original attacker position -> recommended',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{SP}/tactic_recommendation.png', dpi=150)

    # ---- gallery: one example per observed delivery zone -------------------
    # The recommended target is the SAME point every time -- the per-corner
    # outcome is not predictable from the defence (shot AUC 0.557, xG R^2
    # negative out of sample), so a conditional target would be fitting noise.
    # What varies across these panels is the starting point: how far the ball
    # has to move, and how much the shape changes with it.
    zones = df.obs_zone.value_counts().head(6).index.tolist()
    gpicks = []
    for z in zones:
        sub = df[(df.obs_zone == z) & (df.gain_honest > 0)] \
            .sort_values('gain_honest', ascending=False)
        if len(sub):
            gpicks.append(sub.iloc[0])

    nrow = int(np.ceil(len(gpicks) / 2))
    fig2, ax2 = plt.subplots(nrow, 4, figsize=(19, 5.0 * nrow))
    ax2 = np.atleast_2d(ax2)
    for k, pk in enumerate(gpicks):
        rr, cc = divmod(k, 2)
        cid = pk.corner_id
        r = dl.loc[cid]
        tech = r.technique if r.technique in (
            'Inswinging', 'Outswinging', 'Straight') else None
        obs = reconstruct((r.start_x, r.start_y), (r.end_x, r.end_y),
                          r.duration, height_class=r.height_class,
                          technique=tech, end_height=r.end_height)
        rec = reconstruct((r.start_x, r.start_y), (tx, ty), tdur,
                          height_class='High Pass', technique=tech,
                          end_height=r.end_height)
        o = dp_orig[dp_orig.corner_id == cid]
        a = dp_cand[int(pk.best_cand)]
        a = a[a.corner_id == f'{cid}__c{int(pk.best_cand)}']
        po, pa = o[o.teammate & ~o.actor], a[a.teammate & ~a.actor]
        moved = float(np.hypot(r.end_x - tx, r.end_y - ty))
        panel(ax2[rr, cc * 2], o, obs,
              f'as played — {str(pk.technique).lower()}\nto {pk.obs_zone}, '
              f'P={pk.p_ev_original:.2f}')
        panel(ax2[rr, cc * 2 + 1], a, rec,
              f'recommended — same taker, ball to {pk.rec_zone}\n'
              f'target moves {moved:.1f} yd, P={pk.p_ev_selected:.2f}',
              moved=list(zip(po.x, po.y, pa.x, pa.y)), show_target=True)
    for k in range(len(gpicks), nrow * 2):
        rr, cc = divmod(k, 2)
        ax2[rr, cc * 2].axis('off')
        ax2[rr, cc * 2 + 1].axis('off')
    fig2.suptitle(
        'Corner recommendations across the six most common delivery zones\n'
        'The recommended target is the SAME point in every panel (central, '
        '4-9 yd): per-corner outcomes are not predictable from the defence,\n'
        'so a conditional target would be fitting noise. What differs is how '
        'far the ball has to move and how the shape responds.\n'
        'Gold band = the recommended target region; the X is one reconstructed '
        'path into it, not a claim about the exact yard.', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(f'{SP}/tactic_recommendation_gallery.png', dpi=140)

    print(f'\nwrote tactic_recommendation.csv, tactic_recommendation.png '
          f'({len(picks)} examples), tactic_recommendation_gallery.png '
          f'({len(gpicks)} examples)')


if __name__ == '__main__':
    main()

"""The do(shoot) vs do(pass) estimand, restricted to the penalty area.

The positivity audit showed the full-pitch contrast is dominated by states
where nobody ever shoots. Inside the box both actions are genuinely taken, so
the contrast is identified there. This script builds that estimand end to end:

  1. restrict the population to the penalty area (x >= 102, 18 <= y <= 62)
  2. build the confounder set from the 360 freeze frames -- the spatial state
     the identification argument claims closes the backdoor path
  3. fit a propensity model P(shoot | S) and check overlap
  4. trim to common support
  5. contrast naive vs adjusted (IPW + propensity-stratified) effects

This is a BASELINE, not the diffusion model. Its job is to quantify how much
of the raw shoot-vs-pass gap is confounding, and to hand the generative model
a population and an overlap region that are actually identified.

Outcomes: xg_after (xG from this decision to the end of the possession,
inclusive) and goal_after.

PENALTIES ARE EXCLUDED. A penalty is a shot from the spot where passing is not
an option: P(shoot | penalty) = 1, a structural positivity violation of the
same kind as shooting from the own half (positivity.py, section 5, found 248 of
them in this population). Kept in, they are "shots" worth 0.78 xG taken in the
emptiest possible state, which the propensity model cannot recognise as
different -- they inflate the naive contrast and ride into the adjusted one.
`shot_type` comes from StatsBomb's own `shot.type` via stage 2-4.
`--include-penalties` reproduces the earlier, contaminated population.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import PROCESSED

BOX = dict(x_min=102.0, y_min=18.0, y_max=62.0)
POST_A = np.array([120.0, 36.0])
POST_B = np.array([120.0, 44.0])
GOAL = np.array([120.0, 40.0])
TRIM = (0.05, 0.95)


def load_box_decisions(include_penalties=False):
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet')
    if 'shot_type' not in dp:
        raise SystemExit('decision_points.parquet has no shot_type column -- '
                         'rerun stage234_possessions.py and stage5')
    box = dp[(dp.x >= BOX['x_min']) & (dp.y >= BOX['y_min'])
             & (dp.y <= BOX['y_max'])].copy()
    box = box[box.action_type.isin(['shot', 'pass'])].copy()
    pen = box.shot_type == 'Penalty'
    print(f'penalties in the population: {int(pen.sum())} '
          f'({"KEPT -- the earlier, contaminated estimand" if include_penalties else "dropped"})')
    if not include_penalties:
        box = box[~pen].copy()
    box['treat'] = (box.action_type == 'shot').astype(int)
    return box


def freeze_frame_covariates(uuids):
    """Spatial confounders per decision, from the 360 freeze frame."""
    want = set(uuids)
    pf = pq.ParquetFile(f'{PROCESSED}/freeze_frames.parquet')
    keep = []
    for batch in pf.iter_batches(batch_size=500_000,
                                 columns=['uuid', 'x', 'y', 'teammate',
                                          'actor', 'keeper']):
        d = batch.to_pandas()
        d = d[d.uuid.isin(want)]
        if len(d):
            keep.append(d)
    ff = pd.concat(keep, ignore_index=True)
    return ff


def cone_counts(ff, ball):
    """Opponents inside the ball->posts triangle, and other pressure measures.

    The shot cone is the region a defender must occupy to block a direct shot,
    so it is the spatial quantity that drives BOTH the choice to shoot and the
    chance the shot succeeds -- i.e. the confounder the freeze frame supplies.
    """
    rows = []
    for uuid, grp in ff.groupby('uuid', sort=False):
        b = ball.loc[uuid]
        bx, by = float(b.x), float(b.y)
        opp = grp[~grp.teammate]
        ox = opp.x.to_numpy(dtype=float)
        oy = opp.y.to_numpy(dtype=float)

        # inside triangle (ball, post A, post B) via consistent cross products
        def side(px, py, ax, ay, bx_, by_):
            return (bx_ - ax) * (py - ay) - (by_ - ay) * (px - ax)

        d1 = side(ox, oy, bx, by, POST_A[0], POST_A[1])
        d2 = side(ox, oy, POST_A[0], POST_A[1], POST_B[0], POST_B[1])
        d3 = side(ox, oy, POST_B[0], POST_B[1], bx, by)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        in_cone = int(np.sum(~(neg & pos)))

        dists = np.hypot(ox - bx, oy - by) if len(ox) else np.array([np.inf])
        keeper = grp[(~grp.teammate) & grp.keeper]
        rows.append({
            'uuid': uuid,
            'n_in_cone': in_cone,
            'n_goalside': int(np.sum(ox > bx)),
            'nearest_opp': float(dists.min()) if len(ox) else 12.0,
            'n_opp': int(len(ox)),
            'n_team': int(np.sum(grp.teammate & ~grp.actor)),
            'keeper_dist': (float(np.hypot(keeper.x.iloc[0] - GOAL[0],
                                           keeper.y.iloc[0] - GOAL[1]))
                            if len(keeper) else np.nan),
        })
    return pd.DataFrame(rows)


def add_geometry(df):
    dx = GOAL[0] - df.x
    dy = GOAL[1] - df.y
    df['dist_goal'] = np.hypot(dx, dy)
    a = np.hypot(df.x - POST_A[0], df.y - POST_A[1])
    b = np.hypot(df.x - POST_B[0], df.y - POST_B[1])
    cos = np.clip((a ** 2 + b ** 2 - 64.0) / (2 * a * b), -1, 1)
    df['goal_angle'] = np.degrees(np.arccos(cos))
    return df


FEATURES = ['dist_goal', 'goal_angle', 'n_in_cone', 'n_goalside',
            'nearest_opp', 'n_opp', 'n_team', 'keeper_dist', 'under_pressure']


def fit_ps(df):
    """Propensity P(shoot | freeze-frame state), standardized features."""
    X = df[FEATURES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    y = df.treat.to_numpy()
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    return lr.predict_proba(X)[:, 1], lr, X, y


def effects(df, outcome):
    """Naive, IPW and propensity-stratified contrasts on a trimmed frame."""
    t = df.treat.to_numpy()
    yv = df[outcome].to_numpy(dtype=float)
    ps = df.ps.to_numpy()
    naive = yv[t == 1].mean() - yv[t == 0].mean()
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    ipw = ((w * yv * t).sum() / (w * t).sum()
           - (w * yv * (1 - t)).sum() / (w * (1 - t)).sum())
    bins = pd.qcut(df.ps, 10, duplicates='drop')
    est, wts = [], []
    for _, g in df.groupby(bins, observed=True):
        if g.treat.nunique() < 2:
            continue
        est.append(g.loc[g.treat == 1, outcome].mean()
                   - g.loc[g.treat == 0, outcome].mean())
        wts.append(len(g))
    strat = float(np.average(est, weights=wts)) if est else np.nan
    return naive, ipw, strat, len(est)


def cluster_bootstrap(df, outcome, n_boot=200, seed=0):
    """Resample whole MATCHES: decisions within a match are not independent."""
    rng = np.random.default_rng(seed)
    matches = df.match_id.unique()
    by_match = {m: g for m, g in df.groupby('match_id')}
    out = {'naive': [], 'ipw': [], 'strat': []}
    for _ in range(n_boot):
        pick = rng.choice(matches, size=len(matches), replace=True)
        rep = pd.concat([by_match[m] for m in pick], ignore_index=True)
        if rep.treat.nunique() < 2:
            continue
        try:
            rep = rep.copy()
            rep['ps'], *_ = fit_ps(rep)
            rep = rep[(rep.ps > TRIM[0]) & (rep.ps < TRIM[1])]
            n, i, s, _ = effects(rep, outcome)
        except Exception:
            continue
        out['naive'].append(n)
        out['ipw'].append(i)
        out['strat'].append(s)
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
            for k, v in out.items() if v}


def report(df, tag):
    s = df[df.treat == 1]
    p = df[df.treat == 0]
    return {
        'population': tag, 'n': len(df), 'n_shot': len(s), 'n_pass': len(p),
        'xg_shoot': s.xg_after.mean(), 'xg_pass': p.xg_after.mean(),
        'goal_shoot': s.goal_after.mean(), 'goal_pass': p.goal_after.mean(),
    }


def main():
    box = load_box_decisions('--include-penalties' in sys.argv)
    print(f'=== 1. population: the penalty area ===')
    print(f'x >= {BOX["x_min"]:.0f}, {BOX["y_min"]:.0f} <= y <= {BOX["y_max"]:.0f}'
          f'   shots {int((box.treat==1).sum()):,}  passes {int((box.treat==0).sum()):,}')
    print(f'raw P(shoot | in box, action in shoot/pass) = {box.treat.mean():.3f}')

    print('\n=== 2. confounders from the 360 freeze frame ===')
    ff = freeze_frame_covariates(box.uuid)
    ball = box.set_index('uuid')[['x', 'y']]
    cov = cone_counts(ff, ball)
    df = box.merge(cov, on='uuid', how='inner')
    df = add_geometry(df)
    df['keeper_dist'] = df.keeper_dist.fillna(df.keeper_dist.median())
    df['under_pressure'] = df.under_pressure.astype(int)
    print(f'decisions with a usable freeze frame: {len(df):,} of {len(box):,}')
    print(df.groupby('treat')[['dist_goal', 'goal_angle', 'n_in_cone',
                               'nearest_opp', 'n_goalside']].mean()
          .rename(index={0: 'pass', 1: 'shoot'}).round(2).to_string())
    print('  -> shooters are closer, with a wider angle and fewer bodies in the')
    print('     cone. That gap IS the confounding the naive contrast absorbs.')

    print('\n=== 3. propensity model and overlap ===')
    df['ps'], lr, X, y = fit_ps(df)
    print(f'in-sample AUC {roc_auc_score(y, df.ps):.3f}, '
          f'accuracy {lr.score(X, y):.3f}')
    for lo, hi in [(0, .05), (.05, .2), (.2, .8), (.8, .95), (.95, 1.0)]:
        m = (df.ps >= lo) & (df.ps < hi)
        print(f'  propensity [{lo:.2f},{hi:.2f}): {m.sum():>6,} decisions '
              f'({m.mean()*100:>4.1f}%)  shots {int(df.treat[m].sum()):>5,}')

    print(f'\n=== 4. trim to common support {TRIM} ===')
    tr = df[(df.ps > TRIM[0]) & (df.ps < TRIM[1])].copy()
    print(f'kept {len(tr):,} of {len(df):,} ({len(tr)/len(df)*100:.1f}%)  '
          f'shots {int((tr.treat==1).sum()):,}  passes {int((tr.treat==0).sum()):,}')

    print('\n=== 5. contrast: value of shooting vs passing ===')
    rows = [report(df, 'box, untrimmed'), report(tr, 'box, trimmed')]
    r = pd.DataFrame(rows)
    r['naive_xg_diff'] = r.xg_shoot - r.xg_pass
    r['naive_goal_diff'] = r.goal_shoot - r.goal_pass
    print(r.round(4).to_string(index=False))

    for outcome in ['xg_after', 'goal_after']:
        naive, ipw, strat, n_strata = effects(tr, outcome)
        ci = cluster_bootstrap(df, outcome)
        print(f'\n  {outcome}   (95% CI: bootstrap over {df.match_id.nunique()} '
              f'matches, resampling whole matches)')
        print(f'    naive difference      {naive:+.4f}  '
              f'[{ci["naive"][0]:+.4f}, {ci["naive"][1]:+.4f}]')
        print(f'    IPW adjusted          {ipw:+.4f}  '
              f'[{ci["ipw"][0]:+.4f}, {ci["ipw"][1]:+.4f}]   '
              f'({(1-ipw/naive)*100:.0f}% of the raw gap was confounding)')
        print(f'    stratified (deciles)  {strat:+.4f}  '
              f'[{ci["strat"][0]:+.4f}, {ci["strat"][1]:+.4f}]   '
              f'over {n_strata} strata')

    print('\n  Caveat: adjustment uses summary features of the freeze frame, not')
    print('  the full configuration, and 360 frames average ~16 of 22 players.')
    print('  Unobserved: finishing skill, fatigue, off-camera defenders.')


if __name__ == '__main__':
    main()

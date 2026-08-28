"""How to quantify the defensive setup at a corner, and whether it matters.

The delivery-target analysis described the defence with one number: how many
visible outfield defenders stand in the six-yard box. That is enough to ask
"crowded or not" and nothing else -- a flat line on the six-yard line and three
men on the posts with two zonal score identically.

This builds a richer description from the same freeze frame and asks two
questions:

  1. do these descriptors carry information the count does not?
  2. does the best delivery target actually depend on any of them?

The second is the one that matters. The crude split found no conditional
effect, but absence of an effect under a bad measure is weak evidence.

DEFENSIVE MARKING SCHEME. The tactically central choice at a corner is zonal
versus man-marking, and it is not recorded anywhere in the data. It is however
inferable from geometry: under man-marking each defender stands next to an
attacker, under zonal they occupy space regardless of where attackers are. So
the mean distance from a defender to their nearest attacker separates the two
schemes without either being labelled.

VISIBILITY. 360 frames only contain players in camera view. At corners the
camera is on the box, so defender coverage is near complete (median 10 of 10
outfield defenders), and corr(visible, in-box) is 0.234 -- but counts are still
reported as fractions of visible defenders where that makes sense, so the
measures do not silently track camera framing.

Outputs: defense_features.parquet, defense_features.csv
"""
import os

import numpy as np
import pandas as pd

import outdirs  # noqa: F401 -- creates results/ figures/ models/
SP = os.path.dirname(os.path.abspath(__file__))
PROCESSED = f'{SP}/../data/processed'

GOAL = np.array([120.0, 40.0])
NEAR_POST = np.array([120.0, 36.0])
FAR_POST = np.array([120.0, 44.0])
MARK_YD = 2.0          # within this of an attacker counts as marking them
POST_YD = 3.0


def per_corner(g):
    """Defensive descriptors for one corner's freeze frame."""
    d = g[(~g.teammate) & (~g.keeper)]
    a = g[g.teammate & (~g.actor)]
    k = g[g.keeper & (~g.teammate)]
    n = len(d)
    if n == 0 or len(a) == 0:
        return None
    dx, dy = d.x.to_numpy(float), d.y.to_numpy(float)
    ax, ay = a.x.to_numpy(float), a.y.to_numpy(float)

    # distance from each defender to the nearest attacker: low = man-marking,
    # high = zonal (defenders occupying space, not bodies)
    D = np.hypot(dx[:, None] - ax[None, :], dy[:, None] - ay[None, :])
    nearest = D.min(axis=1)

    in6 = ((dx >= 114) & (dy >= 30) & (dy <= 50))
    return {
        'n_def_visible': n,
        'def_in6': int(in6.sum()),
        'def_in6_frac': float(in6.mean()),
        'mark_dist': float(nearest.mean()),
        'marked_frac': float((nearest < MARK_YD).mean()),
        'def_line_x': float(dx.mean()),
        'def_deepest_x': float(dx.max()),
        'def_spread_y': float(dy.std()),
        'near_post_cover': int((np.hypot(dx - NEAR_POST[0],
                                         dy - NEAR_POST[1]) < POST_YD).sum()),
        'far_post_cover': int((np.hypot(dx - FAR_POST[0],
                                        dy - FAR_POST[1]) < POST_YD).sum()),
        'gk_depth': (float(120.0 - k.x.iloc[0]) if len(k) else np.nan),
        'gk_y_off': (float(abs(k.y.iloc[0] - 40.0)) if len(k) else np.nan),
    }


def build():
    pl = pd.read_parquet(f'{SP}/corner_players.parquet')
    rows = {}
    for cid, g in pl.groupby('corner_id', sort=False):
        r = per_corner(g)
        if r is not None:
            rows[cid] = r
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'corner_id'
    return df


def main():
    df = build()
    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet')
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet',
                         columns=['uuid', 'xg_after', 'goal_after'])
    c = pd.read_parquet(f'{SP}/corners.parquet')
    j = (df.reset_index()
           .merge(dl[['corner_id', 'match_id', 'end_x', 'end_y', 'start_x',
                      'start_y', 'technique']], on='corner_id')
           .merge(c[['corner_id', 'shot']], on='corner_id')
           .merge(dp.rename(columns={'uuid': 'corner_id'}), on='corner_id'))
    j['chord'] = np.hypot(j.end_x - j.start_x, j.end_y - j.start_y)
    j = j[j.chord >= 15.0].copy()          # long corners only
    j['lateral'] = np.select(
        [j.end_y < 36, j.end_y < 44, j.end_y < 52],
        ['near post', 'central', 'far post'], default='beyond far post')
    j.to_parquet(f'{SP}/defense_features.parquet', index=False)

    FEATS = ['def_in6', 'def_in6_frac', 'mark_dist', 'marked_frac',
             'def_line_x', 'def_spread_y', 'near_post_cover',
             'far_post_cover', 'gk_depth']
    print(f'{len(j):,} long corners with a defensive description\n')
    print('=== 1. what the descriptors look like ===')
    print(j[FEATS].describe().loc[['mean', 'std', '25%', '50%', '75%']]
          .round(2).to_string())

    print('\n=== 2. are they redundant with the crude count? ===')
    cor = j[FEATS].corr()['def_in6'].drop('def_in6').sort_values(key=abs,
                                                                ascending=False)
    print('correlation with def_in6:')
    print(cor.round(3).to_string())
    print('  (|r| well under 1 means these carry information the count does not)')

    print('\n=== 3. marking scheme: is it bimodal? ===')
    for q in [5, 25, 50, 75, 95]:
        print(f'  mark_dist p{q:<3} {np.percentile(j.mark_dist, q):.2f} yd')
    print(f'  fraction of defenders within {MARK_YD:.0f} yd of an attacker: '
          f'mean {j.marked_frac.mean():.2f}')

    print('\n=== 4. does corner value vary with the defensive setup? ===')
    print(f'{"descriptor":>16}  {"low tercile":>12}  {"mid":>8}  '
          f'{"high":>8}   (mean xG)')
    for f in FEATS:
        if j[f].nunique() < 3:
            continue
        try:
            t = pd.qcut(j[f], 3, labels=['low', 'mid', 'high'], duplicates='drop')
        except ValueError:
            continue
        m = j.groupby(t, observed=True).xg_after.mean()
        if len(m) == 3:
            print(f'{f:>16}  {m.iloc[0]:>12.4f}  {m.iloc[1]:>8.4f}  '
                  f'{m.iloc[2]:>8.4f}')

    print('\n=== 5. THE question: does the best target depend on the defence? ===')
    for f in ['mark_dist', 'def_in6', 'near_post_cover', 'def_line_x']:
        try:
            t = pd.qcut(j[f], 2, labels=['low', 'high'], duplicates='drop')
        except ValueError:
            continue
        print(f'\n  split on {f}:')
        piv = j.pivot_table(index='lateral', columns=t, values='xg_after',
                            aggfunc=['mean', 'size'])
        piv.columns = [f'{a}_{b}' for a, b in piv.columns]
        print(piv.round(4).to_string())
        for side in ['low', 'high']:
            col = f'mean_{side}'
            if col in piv:
                sub = piv[piv[f'size_{side}'] >= 60]
                if len(sub):
                    print(f'    best band when {f} is {side}: '
                          f'{sub[col].idxmax()} ({sub[col].max():.4f})')

    j[['corner_id'] + FEATS].to_csv(f'{SP}/results/defense_features.csv', index=False)
    print('\nwrote defense_features.parquet, results/defense_features.csv')


if __name__ == '__main__':
    main()

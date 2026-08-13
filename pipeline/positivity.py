"""Positivity / overlap audit for the interventional estimand.

E[Y | do(A=a), S=s] is only identified where P(A=a | S=s) > 0. Where an action
is never taken in a state, the data carries no evidence about it and any value
the model returns is extrapolation. A conditional generative model will not
complain -- it samples a plausible-looking continuation regardless -- so the
violation has to be measured up front, not discovered in the output.

Reports, over the 360 causal core:
  1. support per (state bin, action type)
  2. empirical propensities and how much mass sits near zero
  3. what happens to support as the action space is refined to target zones
  4. the concrete shoot-vs-pass contrast, by pitch region
"""
import numpy as np
import pandas as pd

from common import PROCESSED, ZONE_ROWS

MIN_SUPPORT = 10          # observations of an action in a bin to call it supported
ACTIONS = ['pass', 'carry', 'cross', 'dribble', 'shot']


def zone_label(z):
    return f'x{(z // ZONE_ROWS) * 10}-{(z // ZONE_ROWS) * 10 + 10}, ' \
           f'y{(z % ZONE_ROWS) * 10}-{(z % ZONE_ROWS) * 10 + 10}'


def main():
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet',
                         columns=['x', 'y', 'action_type', 'action_zone',
                                  'under_pressure', 'n_opponents_visible',
                                  'xg_after', 'goal_after'])
    n = len(dp)
    print(f'decision points in the 360 causal core: {n:,}\n')

    # ---- state binning: where the ball is, and whether under pressure ----
    dp['sx'] = np.clip((dp.x // 10).astype(int), 0, 11)
    dp['sy'] = np.clip((dp.y // 10).astype(int), 0, 7)
    # integer state id: ball zone (12x8) crossed with under_pressure
    dp['state'] = (dp.sx * 8 + dp.sy) * 2 + dp.under_pressure.astype(int)
    dp['action_type'] = dp['action_type'].astype('object')

    tab = pd.crosstab(dp['state'], dp['action_type'])
    for a in ACTIONS:
        if a not in tab:
            tab[a] = 0
    tab = tab[ACTIONS]
    sizes = tab.sum(axis=1)

    print('=== 1. support per (state bin, action) ===')
    print(f'state bins occupied: {len(tab):,} of {12*8*2} possible')
    print(f'median decisions per bin: {sizes.median():.0f}\n')
    print(f'{"action":>9}  {"bins with 0":>11}  {"bins <10":>9}  '
          f'{"share of decisions in a zero bin":>32}')
    for a in ACTIONS:
        zero = tab[a] == 0
        thin = tab[a] < MIN_SUPPORT
        mass = sizes[zero].sum() / n
        print(f'{a:>9}  {zero.sum():>11,}  {thin.sum():>9,}  {mass*100:>31.1f}%')

    full = ((tab >= MIN_SUPPORT).all(axis=1))
    print(f'\nbins where ALL five actions have >={MIN_SUPPORT} obs: '
          f'{full.sum():,} of {len(tab):,} '
          f'({sizes[full].sum()/n*100:.1f}% of decisions)')

    # ---- propensities ----
    print('\n=== 2. empirical propensity P(A=a | S) ===')
    prop = tab.div(sizes, axis=0)
    w = sizes / n
    print(f'{"action":>9}  {"mean P":>7}  {"P<0.01":>18}  {"P=0":>18}')
    for a in ACTIONS:
        mean_p = float((prop[a] * w).sum())
        near0 = float(w[prop[a] < 0.01].sum())
        exact0 = float(w[prop[a] == 0].sum())
        print(f'{a:>9}  {mean_p:>7.3f}  {near0*100:>17.1f}%  {exact0*100:>17.1f}%')
    print('  (percentages are the share of decisions sitting in such a state)')

    # ---- refining the action space ----
    print('\n=== 3. cost of refining the action space ===')
    print(f'{"action space":>34}  {"cells":>9}  {"observed":>9}  {"supported":>10}')
    acode = dp['action_type'].map({a: i for i, a in enumerate(ACTIONS)})
    spaces = {
        'action type only (5)': acode,
        'type x target zone (5 x 96)':
            acode * 200 + dp['action_zone'].astype(int) + 1,
    }
    for name, col in spaces.items():
        cells = pd.crosstab(dp['state'], col)
        total = len(tab) * col.nunique()
        observed = int((cells > 0).sum().sum())
        supported = int((cells >= MIN_SUPPORT).sum().sum())
        print(f'{name:>34}  {total:>9,}  {observed:>9,}  {supported:>10,}')
    print('  cells = state bins x actions. "supported" = the cells a')
    print(f'  do(a) query can actually be answered from ({MIN_SUPPORT}+ obs).')

    # ---- the headline contrast ----
    print('\n=== 4. do(shoot) vs do(pass), by distance from goal ===')
    dp['band'] = pd.cut(dp.x, [0, 60, 90, 102, 114, 121],
                        labels=['own half', '60-90', '90-102 (edge of box)',
                                '102-114 (box)', '114-120 (6-yd)'])
    g = dp.groupby('band', observed=True)
    out = pd.DataFrame({
        'decisions': g.size(),
        'P(shoot)': g['action_type'].apply(lambda s: (s == 'shot').mean()),
        'P(pass)': g['action_type'].apply(lambda s: (s == 'pass').mean()),
        'mean xg_after': g['xg_after'].mean(),
    })
    out['P(shoot)'] = out['P(shoot)'].map(lambda v: f'{v*100:.2f}%')
    out['P(pass)'] = out['P(pass)'].map(lambda v: f'{v*100:.1f}%')
    out['mean xg_after'] = out['mean xg_after'].round(4)
    out['decisions'] = out['decisions'].map('{:,}'.format)
    print(out.to_string())
    print('\n  do(shoot) in the own half is a structural violation, not a data')
    print('  shortage: nobody shoots from there because it is not a real option.')
    print('  Averaging a do(shoot) effect over all states silently includes it.')


if __name__ == '__main__':
    main()

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
  5. the state refined by the freeze frame -- opponents near the ball -- with
     the camera polygon deciding when that count is a real zero and when it
     is unobserved, and what the refinement costs in support
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import PROCESSED, ZONE_ROWS
from visibility import disc_visible_frac, load_polygons

MIN_SUPPORT = 10          # observations of an action in a bin to call it supported
ACTIONS = ['pass', 'carry', 'cross', 'dribble', 'shot']


def zone_label(z):
    return f'x{(z // ZONE_ROWS) * 10}-{(z // ZONE_ROWS) * 10 + 10}, ' \
           f'y{(z % ZONE_ROWS) * 10}-{(z % ZONE_ROWS) * 10 + 10}'


def main():
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet',
                         columns=['uuid', 'x', 'y', 'action_type', 'action_zone',
                                  'play_pattern',
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

    local_density_section(dp, tab)


LOCAL_R = 5.0         # yd around the ball
SEEN = 0.9            # share of that disc inside the camera polygon to trust a count
LOCAL_LEVELS = ['0 opp', '1 opp', '2+ opp', 'unobserved']


def local_density(dp):
    """Opponents within LOCAL_R of the ball, and whether a low count is real.

    A count of 2+ is known even when part of the disc is off camera (it can
    only be higher). A count of 0 or 1 is only trusted when the camera saw
    at least SEEN of the disc; otherwise it is 'unobserved', not zero.
    """
    want = set(dp.uuid)
    cnt = pd.Series(0, index=dp.uuid.to_numpy())
    ball = dp.set_index('uuid')[['x', 'y']]
    for b in pq.ParquetFile(f'{PROCESSED}/freeze_frames.parquet').iter_batches(
            batch_size=2_000_000, columns=['uuid', 'x', 'y', 'teammate']):
        f = b.to_pandas()
        f = f[~f.teammate & f.uuid.isin(want)]
        bx = ball.x.reindex(f.uuid).to_numpy()
        by = ball.y.reindex(f.uuid).to_numpy()
        near = np.hypot(f.x.to_numpy() - bx, f.y.to_numpy() - by) < LOCAL_R
        add = f[near].groupby('uuid').size()
        cnt.loc[add.index] += add.to_numpy()
    polys = load_polygons(f'{PROCESSED}/visible_area.parquet')
    vis = np.array([disc_visible_frac(polys.get(u), x, y, LOCAL_R)
                    for u, x, y in zip(dp.uuid, dp.x, dp.y)])
    n = cnt.reindex(dp.uuid).to_numpy()
    level = np.select([n >= 2, vis < SEEN, n == 1], ['2+ opp', 'unobserved', '1 opp'],
                      default='0 opp')
    return n, vis, level


def local_density_section(dp, coarse_tab):
    print(f'\n=== 5. refining the state with the freeze frame: opponents within '
          f'{LOCAL_R:.0f} yd of the ball ===')
    n, vis, level = local_density(dp)
    dp['local'] = level
    naive_zero = (n == 0)
    print(f'a raw count of 0 on {naive_zero.mean()*100:.1f}% of decisions; for '
          f'{(naive_zero & (vis < SEEN)).sum() / naive_zero.sum()*100:.1f}% of '
          f'those the camera saw less than {SEEN:.0%} of the disc,')
    print('so reading the freeze frame alone would call them "unpressured" when '
          'they are unobserved.')
    band = pd.cut(dp.x, [0, 60, 90, 102, 121],
                  labels=['own half', '60-90', '90-102', '102-120 (box)'])
    share = pd.crosstab(band, dp.local, normalize='index')[LOCAL_LEVELS] * 100
    print('\nlocal state by pitch band (% of decisions):')
    print(share.round(1).to_string())

    lcode = dp.local.map({k: i for i, k in enumerate(LOCAL_LEVELS)})
    dp['state_ff'] = dp.state * 4 + lcode
    tab = pd.crosstab(dp.state_ff, dp.action_type)
    for a in ACTIONS:
        if a not in tab:
            tab[a] = 0
    tab = tab[ACTIONS]
    sizes = tab.sum(axis=1)
    total = len(dp)
    print(f'\nstate bins: {len(coarse_tab)} (zone x pressure) -> {len(tab)} '
          f'(x local density); median decisions per bin '
          f'{coarse_tab.sum(axis=1).median():.0f} -> {sizes.median():.0f}')
    print(f'{"action":>9}  {"zero-bin mass, coarse":>22}  {"refined":>8}  '
          f'{"<10 obs, coarse":>16}  {"refined":>8}')
    cs = coarse_tab.sum(axis=1)
    for a in ACTIONS:
        z0 = cs[coarse_tab[a] == 0].sum() / total * 100
        z1 = sizes[tab[a] == 0].sum() / total * 100
        t0 = cs[coarse_tab[a] < MIN_SUPPORT].sum() / total * 100
        t1 = sizes[tab[a] < MIN_SUPPORT].sum() / total * 100
        print(f'{a:>9}  {z0:>21.1f}%  {z1:>7.1f}%  {t0:>15.1f}%  {t1:>7.1f}%')
    print('  (share of all decisions sitting in a bin where the action has 0 /')
    print(f'   fewer than {MIN_SUPPORT} observations)')

    box = dp[(dp.x >= 102) & (dp.y >= 18) & (dp.y <= 62)
             & dp.action_type.isin(['shot', 'pass'])]
    g = box.groupby('local')
    out = pd.DataFrame({'decisions': g.size(),
                        'P(shoot | shoot or pass)': g.action_type.apply(
                            lambda v: (v == 'shot').mean()),
                        'mean xg_after': g.xg_after.mean()}).reindex(LOCAL_LEVELS)
    print('\nthe estimand_box.py population (penalty area, shoot vs pass) by '
          'local density:')
    print(out.round(3).to_string())
    unobs = (box.local == 'unobserved').mean()
    print(f'  {unobs*100:.1f}% of that population has an unobserved local state.')
    ps = out['P(shoot | shoot or pass)']
    if ps.between(0.05, 0.95).all():
        print('  Positivity holds in every local level here -- both actions are')
        print('  taken under every density -- which is what estimand_box.py needs.')
    else:
        print(f'  Positivity FAILS in: {list(ps[~ps.between(0.05, 0.95)].index)}')

    # Penalties: a shot from the spot with play pattern 'Other'. Passing is not
    # an option, so they are a structural positivity violation, and their
    # freeze frames have a narrow camera -- they are most of 'unobserved'.
    pen = ((box.action_type == 'shot') & (box.play_pattern == 'Other')
           & ((box.x - 108).abs() <= 0.5) & ((box.y - 40).abs() <= 0.5))
    un = box.local == 'unobserved'
    print(f'\n  penalties in that population: {int(pen.sum())} of {len(box):,} '
          f'({pen.mean()*100:.1f}%), all shots by construction -- P(shoot) = 1,')
    print('  a structural violation like shooting from the own half. They are')
    print(f'  {int((pen & un).sum())} of the {int(un.sum())} unobserved rows (a '
          f'penalty frame has a narrow camera),')
    print('  which is why "unobserved" looks so valuable. Drop them before any')
    print('  do(shoot) vs do(pass) contrast; estimand_box.py currently keeps them.')


if __name__ == '__main__':
    main()

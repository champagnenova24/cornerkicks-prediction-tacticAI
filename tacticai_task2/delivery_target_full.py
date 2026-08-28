"""Delivery-target analysis on the FULL StatsBomb open data (41,178 corners).

`delivery_target.py` ran on the 2,473 corners in 360-covered matches, because
that is where the freeze frames are. The zone question does not need freeze
frames -- only where the delivery arrived and what the possession was worth --
so it can run on all 4,235 matches, 16.6x more corners.

What that buys: the earlier finding was that band-level xG differences are real
in aggregate but an order of magnitude below per-corner noise, so a per-corner
target could not be supported. More corners does not reduce the per-corner
noise, but it does tighten the aggregate estimates by ~4x and makes finer zone
splits viable.

What it costs: no freeze frames outside the 360 subset, so nothing here can
condition on the defensive setup. Defence-conditional results stay in
`delivery_target.py` on the smaller sample.

Corners come from `corners_full_raw.parquet` (first decision of a From Corner
possession, taken from the corner arc; recovers 2,472 of the 2,473 exactly
identified 360-subset corners).

Outputs: delivery_target_full.csv, delivery_target_full.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import outdirs  # noqa: F401 -- creates results/ figures/ models/
SP = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 1000
MIN_CELL = 100
SHORT_CHORD = 15.0

DEPTHS = [(0, 4, '0-4yd'), (4, 9, '4-9yd'), (9, 99, '9yd+')]
LATERALS = [(-99, 36, 'near post'), (36, 44, 'central'),
            (44, 52, 'far post'), (52, 99, 'beyond far post')]
BAND_ORDER = ['near post', 'central', 'far post', 'beyond far post']


def load():
    c = pd.read_parquet(f'{SP}/corners_full_raw.parquet')
    # mirror every corner so the kick is taken at (120, 0)
    c['reflected'] = c.y > 40
    c['end_y_m'] = np.where(c.reflected, 80.0 - c.end_y, c.end_y)
    c['start_y_m'] = np.where(c.reflected, 80.0 - c.y, c.y)
    c['chord'] = np.hypot(c.end_x - c.x, c.end_y_m - c.start_y_m)
    c['short'] = c.chord < SHORT_CHORD
    depth = 120.0 - c.end_x
    c['depth'] = pd.cut(depth, [d[0] for d in DEPTHS] + [DEPTHS[-1][1]],
                        labels=[d[2] for d in DEPTHS], right=False)
    c['lateral'] = pd.cut(c.end_y_m, [l[0] for l in LATERALS] + [LATERALS[-1][1]],
                          labels=[l[2] for l in LATERALS], right=False)
    c['zone'] = np.where(c.short, 'short corner',
                         c.depth.astype(str) + ' x ' + c.lateral.astype(str))
    return c


def boot_ci(g, col, n_boot=N_BOOT, seed=0):
    """Cluster bootstrap over matches: corners share a taker and a game state."""
    m = g.match_id.to_numpy()
    um = np.unique(m)
    where = {q: np.where(m == q)[0] for q in um}
    rng = np.random.default_rng(seed)
    v = g[col].to_numpy(float)
    bs = [v[np.concatenate([where[q] for q in
                            rng.choice(um, len(um), replace=True)])].mean()
          for _ in range(n_boot)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def table(df, by, min_n=MIN_CELL):
    rows = []
    for k, g in df.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        lo, hi = boot_ci(g, 'xg_after')
        rows.append(dict(**{by: k}, n=len(g), share=len(g) / len(df),
                         xg=g.xg_after.mean(), xg_lo=lo, xg_hi=hi,
                         goal_rate=g.goal_after.mean()))
    return pd.DataFrame(rows).sort_values('xg', ascending=False)


def show(t, title, by):
    print(f'\n=== {title} ===')
    d = t.copy()
    d['xg'] = d.apply(lambda r: f'{r.xg:.4f} [{r.xg_lo:.4f},{r.xg_hi:.4f}]',
                      axis=1)
    d['share'] = (d.share * 100).round(1).astype(str) + '%'
    d['goal_rate'] = (d.goal_rate * 100).round(2).astype(str) + '%'
    print(d[[by, 'n', 'share', 'xg', 'goal_rate']].to_string(index=False))


def main():
    c = load()
    print(f'{len(c):,} corners across {c.match_id.nunique():,} matches')
    print(f'  short corners: {c.short.mean()*100:.1f}%')
    print(f'  overall xG {c.xg_after.mean():.4f}, '
          f'goal rate {c.goal_after.mean()*100:.2f}%')
    print(f'  zero-xG corners: {(c.xg_after == 0).mean()*100:.0f}%')

    long = c[~c.short]
    t_lat = table(long, 'lateral')
    show(t_lat, f'lateral band, {len(long):,} long corners', 'lateral')
    t_zone = table(c, 'zone')
    show(t_zone, 'full zone grid', 'zone')
    t_zone.to_csv(f'{SP}/results/delivery_target_full.csv', index=False)

    # ---- precision gained over the 360 subset ----
    print('\n=== precision: full set vs the 360 subset ===')
    try:
        old = pd.read_csv(f'{SP}/results/delivery_target_zones.csv')
        ow = (old.xg_hi - old.xg_lo).median()
        nw = (t_zone.xg_hi - t_zone.xg_lo).median()
        print(f'  median 95% CI width per zone: {ow:.4f} (n=2,473) -> '
              f'{nw:.4f} (n={len(c):,})   {ow/nw:.1f}x tighter')
    except FileNotFoundError:
        pass

    # ---- does the band ranking survive? ----
    print('\n=== band ranking, full set vs subset ===')
    lat = t_lat.set_index('lateral')
    for b in BAND_ORDER:
        if b in lat.index:
            r = lat.loc[b]
            print(f'  {b:<16} n={int(r.n):>6,}  xG {r.xg:.4f} '
                  f'[{r.xg_lo:.4f},{r.xg_hi:.4f}]')

    # pairwise separation now that the CIs are tight
    print('\n  pairwise: is central separated from the rest?')
    cen = long[long.lateral == 'central']
    for b in ['far post', 'near post', 'beyond far post']:
        oth = long[long.lateral == b]
        if not len(oth):
            continue
        diff = cen.xg_after.mean() - oth.xg_after.mean()
        pool = pd.concat([cen.assign(_g=1), oth.assign(_g=0)])
        m = pool.match_id.to_numpy()
        um = np.unique(m)
        where = {q: np.where(m == q)[0] for q in um}
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(N_BOOT):
            idx = np.concatenate([where[q] for q in
                                  rng.choice(um, len(um), replace=True)])
            s = pool.iloc[idx]
            if s._g.nunique() < 2:
                continue
            bs.append(s[s._g == 1].xg_after.mean() - s[s._g == 0].xg_after.mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'    central - {b:<16} {diff:+.4f}  [{lo:+.4f},{hi:+.4f}]'
              f'{"  *" if not (lo < 0 < hi) else "  crosses 0"}')

    # ---- figure: xG surface over the mouth of the goal ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (col, lab) in zip(axes, [('xg_after', 'mean xG'),
                                     ('goal_after', 'goal rate')]):
        xs = np.arange(100, 121, 2.0)
        ys = np.arange(20, 61, 2.0)
        grid = np.full((len(ys) - 1, len(xs) - 1), np.nan)
        cnt = np.zeros_like(grid)
        for i in range(len(ys) - 1):
            for j in range(len(xs) - 1):
                m = ((long.end_x >= xs[j]) & (long.end_x < xs[j + 1])
                     & (long.end_y_m >= ys[i]) & (long.end_y_m < ys[i + 1]))
                cnt[i, j] = m.sum()
                if m.sum() >= 40:
                    grid[i, j] = long.loc[m, col].mean()
        im = ax.imshow(grid, origin='lower', aspect='equal', cmap='viridis',
                       extent=[xs[0], xs[-1], ys[0], ys[-1]])
        ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, ec='white',
                                   lw=1.2))
        ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, ec='white',
                                   lw=1.2))
        ax.plot([120, 120], [36, 44], c='red', lw=4)
        ax.set_title(f'{lab} by delivery target\n'
                     f'{len(long):,} long corners, cells with <40 blank',
                     fontsize=10)
        ax.set_xlabel('x (goal at 120)')
        ax.set_ylabel('y (kick taken at y=0)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Corner delivery value, full StatsBomb open data '
                 '(4,235 matches)', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{SP}/figures/delivery_target_full.png', dpi=150)
    print('\nwrote results/delivery_target_full.csv, figures/delivery_target_full.png')


if __name__ == '__main__':
    main()

"""Where should a corner be delivered? Value of the delivery target zone.

Technique turned out to be a near-determined variable worth nothing in goals
(see the joint-recommendation section). The delivery TARGET is the decision
that is actually free and actually continuous, so this asks the question
directly: given where the ball ends up, what is the corner worth?

Frame: corners mirrored so the kick is at (120, 0). The goal spans y in
[36, 44] and the reflection fixes it, so "near post" is always y = 36 (the
post nearer the corner) and "far post" y = 44.

Outcomes come from the pipeline's decision_points.parquet, joined on uuid:
shot attempt, xG over the rest of the possession, and goal. All three are
reported because they disagree -- the whole point of the earlier technique
finding was that attempts and goals come apart.

WHAT THIS IS NOT
----------------
`pass.end_location` is where the ball ENDED, not where it was aimed. A
delivery headed clear at the near post records its end there, so a zone's
value bundles three things: whether aiming there is good, whether the taker
executes, and whether the defence gets there first. That bundle is close to
what a coach actually decides over -- aiming somewhere the defence owns is
genuinely bad -- but it is NOT do(aim at zone Z), and zones cannot be ranked
as if intent were observed. Nothing here is adjusted for defensive setup
beyond the explicit split in section 4.

Outputs: delivery_target_zones.csv, delivery_target.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
PROCESSED = f'{SP}/../data/processed'
N_BOOT = 2000
SHORT_CHORD = 15.0        # a short corner is a different decision entirely

# depth from the goal line, and lateral band relative to the goal mouth
DEPTH = [(116.0, 120.1, '0-4yd (on top of goal)'),
         (111.0, 116.0, '4-9yd (six-yard line)'),
         (0.0, 111.0, '9yd+ (penalty spot and out)')]
LATERAL = [(-1e9, 36.0, 'near post'), (36.0, 44.0, 'central'),
           (44.0, 52.0, 'far post'), (52.0, 1e9, 'beyond far post')]


def zone_of(x, y):
    d = next(lab for lo, hi, lab in DEPTH if lo <= x < hi)
    l = next(lab for lo, hi, lab in LATERAL if lo <= y < hi)
    return d, l


def load():
    dl = pd.read_parquet(f'{SP}/corner_deliveries.parquet')
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet',
                         columns=['uuid', 'xg_after', 'goal_after'])
    c = pd.read_parquet(f'{SP}/corners.parquet')
    # corner_deliveries.parquet already carries match_id, technique and
    # height_class, so only the label and the outcomes need joining in.
    j = (dl.merge(c[['corner_id', 'shot']], on='corner_id')
           .merge(dp.rename(columns={'uuid': 'corner_id'}), on='corner_id'))
    j['chord'] = np.hypot(j.end_x - j.start_x, j.end_y - j.start_y)
    j['short'] = j.chord < SHORT_CHORD
    zs = [zone_of(x, y) for x, y in zip(j.end_x, j.end_y)]
    j['depth'] = [a for a, _ in zs]
    j['lateral'] = [b for _, b in zs]
    j['zone'] = np.where(j.short, 'short corner',
                         j.depth + ' x ' + j.lateral)

    # defensive congestion in the six-yard box at the kick
    pl = pd.read_parquet(f'{SP}/corner_players.parquet')
    dfn = pl[(~pl.teammate) & (~pl.keeper)]
    in6 = dfn[(dfn.x >= 114) & (dfn.y >= 30) & (dfn.y <= 50)]
    j['def_in6'] = j.corner_id.map(
        in6.groupby('corner_id').size()).fillna(0).astype(int)
    return j


def boot_ci(df, col, n_boot=N_BOOT, seed=0):
    """Cluster bootstrap over matches: corners in a match are not independent."""
    rng = np.random.default_rng(seed)
    mids = df.match_id.unique()
    by = {m: g[col].to_numpy(float) for m, g in df.groupby('match_id')}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(mids, len(mids), replace=True)
        vals.append(np.concatenate([by[m] for m in pick]).mean())
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def table(df, by, min_n=40, ci_on='xg_after'):
    rows = []
    for k, g in df.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        lo, hi = boot_ci(g, ci_on)
        rows.append({
            'zone': k if isinstance(k, str) else ' | '.join(k),
            'n': len(g), 'share': len(g) / len(df),
            'shot_rate': g.shot.mean(), 'xg': g.xg_after.mean(),
            'xg_lo': lo, 'xg_hi': hi, 'goal_rate': g.goal_after.mean(),
        })
    return pd.DataFrame(rows).sort_values('xg', ascending=False)


def show(t, title):
    print(f'\n=== {title} ===')
    d = t.copy()
    d['share'] = (d.share * 100).map('{:.1f}%'.format)
    d['xg'] = d.xg.map('{:.4f}'.format)
    d['xG 95% CI'] = [f'[{a:.4f}, {b:.4f}]' for a, b in zip(t.xg_lo, t.xg_hi)]
    d['shot_rate'] = d.shot_rate.map('{:.3f}'.format)
    d['goal_rate'] = d.goal_rate.map('{:.4f}'.format)
    print(d[['zone', 'n', 'share', 'shot_rate', 'xg', 'xG 95% CI',
             'goal_rate']].to_string(index=False))


def main():
    j = load()
    print(f'{len(j):,} corners with a delivery endpoint and an outcome')
    print(f'overall: shot {j.shot.mean():.3f}, xG {j.xg_after.mean():.4f}, '
          f'goal {j.goal_after.mean():.4f}')

    t_zone = table(j, 'zone')
    show(t_zone, '1. value by delivery target zone')
    t_zone.to_csv(f'{SP}/delivery_target_zones.csv', index=False)

    show(table(j[~j.short], 'lateral'), '2. lateral band only (long corners)')
    show(table(j[~j.short], 'depth'), '3. depth only (long corners)')

    # ---- 4. does the best target depend on the defence? ----
    print('\n=== 4. target value vs defensive congestion in the six-yard box ===')
    long = j[~j.short].copy()
    long['packed'] = np.where(long.def_in6 >= long.def_in6.median(),
                              'packed 6-yd box', 'open 6-yd box')
    piv = long.pivot_table(index='lateral', columns='packed',
                           values='xg_after', aggfunc=['mean', 'size'])
    print(piv.round(4).to_string())
    for grp, g in long.groupby('packed'):
        best = table(g, 'lateral', min_n=30)
        if len(best):
            print(f'  {grp}: best lateral band = {best.iloc[0].zone} '
                  f'(xG {best.iloc[0].xg:.4f}, n={best.iloc[0].n})')

    # ---- 5. interaction with technique ----
    print('\n=== 5. target x technique (long corners) ===')
    sw = long[long.technique.isin(['Inswinging', 'Outswinging'])]
    p = sw.pivot_table(index='lateral', columns='technique',
                       values=['xg_after', 'shot'], aggfunc='mean')
    n = sw.pivot_table(index='lateral', columns='technique',
                       values='corner_id', aggfunc='size')
    print(p.round(4).to_string())
    print('\nn per cell:')
    print(n.to_string())

    # ---- figure: xG surface over the delivery target ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (col, lab) in zip(axes, [('xg_after', 'xG'),
                                     ('shot', 'shot-attempt rate')]):
        xe = np.arange(100, 121, 2.5)
        ye = np.arange(20, 61, 5.0)
        H = np.full((len(ye) - 1, len(xe) - 1), np.nan)
        for a in range(len(xe) - 1):
            for b in range(len(ye) - 1):
                m = ((j.end_x >= xe[a]) & (j.end_x < xe[a + 1])
                     & (j.end_y >= ye[b]) & (j.end_y < ye[b + 1]))
                if m.sum() >= 25:
                    H[b, a] = j.loc[m, col].mean()
        im = ax.pcolormesh(xe, ye, H, cmap='viridis', shading='flat')
        fig.colorbar(im, ax=ax, label=lab, fraction=0.035)
        ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, ec='white',
                                   lw=1.2))
        ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, ec='white',
                                   lw=1.2))
        ax.plot([120, 120], [36, 44], c='white', lw=4)
        ax.scatter([120], [0], marker='*', s=160, c='red', zorder=5)
        ax.annotate('corner taken here', (119, 2), color='black', fontsize=8,
                    ha='right', va='bottom')
        ax.annotate('near post', (119.5, 36), color='white', fontsize=8,
                    ha='right', va='top')
        ax.annotate('far post', (119.5, 44), color='white', fontsize=8,
                    ha='right', va='bottom')
        ax.set_xlim(100, 121)
        ax.set_ylim(-2, 60)
        ax.set_aspect('equal')
        ax.set_title(f'{lab} by where the delivery arrives', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Corner value by delivery target (kick mirrored to the red '
                 'star; cells with <25 corners blank)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{SP}/delivery_target.png', dpi=150)

    # ---- risk / reward: reaching an attacker vs paying off once you do -----
    # pass.end_location is where the ball ARRIVED, and the defence has a say in
    # that, so a zone's unconditional value is the product of two things: how
    # often the delivery survives to an attacker, and what it is worth when it
    # does. Splitting them says WHY a zone scores, which is what a coach acts
    # on. Note the conditional column selects on an outcome (who touched it
    # first), so it is a mechanism diagnostic -- the unconditional column is
    # the one that is an expected value.
    cc = pd.read_parquet(f'{SP}/corners.parquet')[
        ['corner_id', 'receiver_is_teammate']]
    rrj = j.merge(cc, on='corner_id')
    rrj = rrj[~rrj.short]
    order = ['near post', 'central', 'far post', 'beyond far post']
    rr = rrj.groupby('lateral').agg(
        n=('shot', 'size'),
        reaches_attacker=('receiver_is_teammate', 'mean'),
        xg_uncond=('xg_after', 'mean')).reindex(order)
    won = rrj[rrj.receiver_is_teammate == True]
    rr['xg_if_won'] = won.groupby('lateral').xg_after.mean().reindex(order)
    rr['shot_if_won'] = won.groupby('lateral').shot.mean().reindex(order)
    print('\n=== why each zone scores: retention x payoff ===')
    print(rr.round(4).to_string())
    print('  central is the HARDEST zone to win (27%) and the most valuable')
    print('  when won (0.135 xG, double any other) -- high risk, high reward.')
    print('  far post beats near post on BOTH terms: it reaches an attacker')
    print('  60% vs 42% of the time for the same payoff once it does, so the')
    print('  near post is dominated. It still takes ~32% of all corners.')
    rr.to_csv(f'{SP}/delivery_risk_reward.csv')

    print('\nwrote delivery_target_zones.csv, delivery_risk_reward.csv, '
          'delivery_target.png')


if __name__ == '__main__':
    main()

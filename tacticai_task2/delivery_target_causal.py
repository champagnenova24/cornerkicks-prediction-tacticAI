"""Causal version of the delivery-target ranking: the corner analog of
`pipeline/estimand_box.py`.

`delivery_target.py` and `delivery_target_full.py` rank zones by the mean xG
of corners that arrived there. That is E[Y | Z = z], not E[Y(z)]: the defence
is set before the ball is struck, the taker aims with it in view, and a zone
where the defence is weak attracts both deliveries and chances. The README
names the missing piece -- "target zone as treatment, freeze-frame geometry as
the adjustment set, positivity checked per zone". This is that script.

    treatment    Z = lateral band of the arrival point (near post / central /
                 far post / beyond far post), long corners only
    adjustment   the freeze frame AT THE KICK, positions only:
                   defence   -- def_in6, marking distance, line depth, spread,
                                post cover, keeper depth/offset, n visible
                   attack    -- attackers in the six-yard box, per lateral
                                band, mean depth, n visible
                 plus the swing (in/out/straight/none), which is fixed before
                 the ball is struck: side x taker's foot determines it
    excluded     velocities (flight-window averages -- they include the run
                 TO the delivery, i.e. they are downstream of the treatment),
                 and pass height (chosen jointly with the target, a
                 co-treatment rather than a confounder)
    outcomes     xg_after, goal_after, shot

Steps, mirroring estimand_box.py:

  1. positivity per zone -- a multinomial generalized propensity e_z(S), and
     the share of corners where each zone is essentially never chosen
  2. positivity by state bins -- zone counts within coarse defensive/attacking
     states, so the empty cells are visible rather than extrapolated over
  3. per contrast (every zone vs central), restrict to the two zones, fit a
     binary propensity, trim to common support, and compare naive vs IPW vs
     cross-fitted AIPW, cluster-bootstrapped over matches with the propensity
     refit in every replicate

WHAT THIS STILL IS NOT. `pass.end_location` is where the ball arrived, so the
treatment is do(arrival zone), not do(aim at zone): the defence winning the
ball short is part of "arrived at the near post". Personnel (who takes, who
attacks the ball) is unobserved, and 360 frames hold only visible players --
at corners that is near-complete for defenders, see defense_features.py. The
adjusted numbers are a better-identified description, not a guarantee.

Outputs: results/delivery_target_causal.csv, results/delivery_target_positivity.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import outdirs  # noqa: F401 -- creates results/ figures/ models/
SP = os.path.dirname(os.path.abspath(__file__))

ZONES = ['near post', 'central', 'far post', 'beyond far post']
REF = 'central'
TRIM = (0.05, 0.95)
EPS_POS = 0.05          # e_z(S) below this = zone z is essentially never chosen
N_BOOT = 500
OUTCOMES = ['xg_after', 'goal_after', 'shot']

DEF_COV = ['def_in6', 'def_in6_frac', 'mark_dist', 'marked_frac', 'def_line_x',
           'def_deepest_x', 'def_spread_y', 'near_post_cover', 'far_post_cover',
           'gk_depth', 'gk_y_off', 'n_def_visible']
ATT_COV = ['att_in6', 'att_near', 'att_central', 'att_far', 'att_beyond',
           'att_mean_x', 'n_att_visible']
TECH_COV = ['tech_in', 'tech_out', 'tech_straight']
COV = DEF_COV + ATT_COV + TECH_COV


def attack_geometry(pl):
    """Attacker positions at the kick, binned by the same lateral bands as
    the treatment -- where the runners start is the obvious confounder of
    where the ball goes."""
    a = pl[pl.teammate & ~pl.actor & ~pl.keeper]
    box = a[(a.x >= 102) & (a.y >= 18) & (a.y <= 62)]
    rows = []
    for cid, g in box.groupby('corner_id', sort=False):
        y = g.y.to_numpy(float)
        rows.append({'corner_id': cid,
                     'att_in6': int(((g.x >= 114) & (y >= 30) & (y <= 50)).sum()),
                     'att_near': int((y < 36).sum()),
                     'att_central': int(((y >= 36) & (y < 44)).sum()),
                     'att_far': int(((y >= 44) & (y < 52)).sum()),
                     'att_beyond': int((y >= 52).sum()),
                     'att_mean_x': float(g.x.mean())})
    out = pd.DataFrame(rows)
    nvis = a.groupby('corner_id').size().rename('n_att_visible').reset_index()
    return out.merge(nvis, on='corner_id', how='right').fillna(
        {'att_in6': 0, 'att_near': 0, 'att_central': 0, 'att_far': 0,
         'att_beyond': 0, 'att_mean_x': 102.0})


def load():
    d = pd.read_parquet(f'{SP}/defense_features.parquet')   # long corners
    pl = pd.read_parquet(f'{SP}/corner_players.parquet')
    d = d.merge(attack_geometry(pl), on='corner_id', how='left')
    tech = d.technique.fillna('none')
    d['tech_in'] = (tech == 'Inswinging').astype(int)
    d['tech_out'] = (tech == 'Outswinging').astype(int)
    d['tech_straight'] = (tech == 'Straight').astype(int)
    for c in ['gk_depth', 'gk_y_off']:
        d[c] = d[c].fillna(d[c].median())
    d[ATT_COV] = d[ATT_COV].fillna(0)
    d['goal_after'] = d.goal_after.astype(float)
    d['shot'] = d.shot.astype(float)
    return d.reset_index(drop=True)


def standardize(X):
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)


# ------------------------------------------------------------ positivity ---
def generalized_propensity(d):
    """Multinomial e_z(S) = P(Z = z | S), out of fold so it is not flattered
    by overfitting."""
    X = standardize(d[COV].to_numpy(float))
    z = d.lateral.map({k: i for i, k in enumerate(ZONES)}).to_numpy()
    E = np.zeros((len(d), len(ZONES)))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = LogisticRegression(max_iter=3000, C=0.5).fit(X[tr], z[tr])
        E[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return E, z


def state_bins(d):
    """Coarse pre-kick state: packed six-yard box x where the attackers are
    concentrated x swing. Few enough cells to read, enough to find holes."""
    packed = np.where(d.def_in6 >= d.def_in6.median(), 'packed', 'open')
    # which side of the goal the runners are loaded on: near-post band vs
    # far-post-and-beyond, with central runners counted for neither
    far = d.att_far + d.att_beyond
    heavy = np.select([d.att_near > far, d.att_near < far],
                      ['att near-loaded', 'att far-loaded'],
                      default='att balanced')
    tech = np.select([d.tech_in == 1, d.tech_out == 1, d.tech_straight == 1],
                     ['in', 'out', 'straight'], default='none')
    return pd.Series([f'{a} | {b} | {c}' for a, b, c in zip(packed, heavy, tech)],
                     index=d.index)


# ------------------------------------------------------------- estimators ---
def fit_ps(X, t):
    return LogisticRegression(max_iter=3000, C=0.5).fit(X, t).predict_proba(X)[:, 1]


def estimates(X, t, y, ps):
    """Naive, Hajek IPW and cross-fitted AIPW on the trimmed sample."""
    keep = (ps > TRIM[0]) & (ps < TRIM[1])
    X, t, y, ps = X[keep], t[keep], y[keep], ps[keep]
    if len(np.unique(t)) < 2 or min(t.sum(), (1 - t).sum()) < 10:
        return dict(naive=np.nan, ipw=np.nan, aipw=np.nan, n=int(keep.sum()))
    naive = y[t == 1].mean() - y[t == 0].mean()
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    ipw = ((w * y * t).sum() / (w * t).sum()
           - (w * y * (1 - t)).sum() / (w * (1 - t)).sum())
    # AIPW: outcome regressions per arm, cross-fitted so a unit's own outcome
    # never informs its own prediction.
    mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        for arm, mu in ((1, mu1), (0, mu0)):
            s = tr[t[tr] == arm]
            mu[te] = Ridge(alpha=10.0).fit(X[s], y[s]).predict(X[te])
    psi = (mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps))
    return dict(naive=naive, ipw=ipw, aipw=float(psi.mean()), n=int(keep.sum()))


def contrast(d, zone, outcome, n_boot=N_BOOT, seed=0):
    sub = d[d.lateral.isin([zone, REF])].reset_index(drop=True)
    X = standardize(sub[COV].to_numpy(float))
    t = (sub.lateral == zone).to_numpy(int)
    y = sub[outcome].to_numpy(float)
    ps = fit_ps(X, t)
    point = estimates(X, t, y, ps)
    point['ps_auc'] = roc_auc_score(t, ps)
    point['trimmed_pct'] = 100 * (1 - point['n'] / len(sub))

    mids = sub.match_id.to_numpy()
    um = np.unique(mids)
    where = {m: np.where(mids == m)[0] for m in um}
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in ('naive', 'ipw', 'aipw')}
    for _ in range(n_boot):
        idx = np.concatenate([where[m] for m in rng.choice(um, len(um))])
        if len(np.unique(t[idx])) < 2:
            continue
        Xb = standardize(sub[COV].to_numpy(float)[idx])
        e = estimates(Xb, t[idx], y[idx], fit_ps(Xb, t[idx]))
        for k in bs:
            if np.isfinite(e[k]):
                bs[k].append(e[k])
    for k, v in bs.items():
        point[f'{k}_lo'], point[f'{k}_hi'] = np.percentile(v, [2.5, 97.5])
    point.update(zone=zone, ref=REF, outcome=outcome, n_zone=int(t.sum()),
                 n_ref=int((1 - t).sum()))
    return point


def main():
    d = load()
    print(f'{len(d):,} long corners in {d.match_id.nunique()} matches, '
          f'{len(COV)} pre-kick covariates')
    print(d.lateral.value_counts().reindex(ZONES).to_string())

    # ---- 1. positivity per zone ------------------------------------------
    print('\n=== 1. positivity per zone: generalized propensity e_z(S) ===')
    E, z = generalized_propensity(d)
    from sklearn.metrics import log_loss
    base = np.bincount(z, minlength=len(ZONES)) / len(z)
    print(f'out-of-fold log loss {log_loss(z, E):.3f} vs {log_loss(z, np.tile(base, (len(z), 1))):.3f}'
          f' for the marginal -- the setup does predict the target')
    prow = []
    print(f'{"zone":<17}{"share":>7}{"e_z p5":>8}{"median":>8}{"p95":>7}'
          f'{"  e_z<" + str(EPS_POS):>11}')
    for k, zn in enumerate(ZONES):
        e = E[:, k]
        low = (e < EPS_POS).mean()
        prow.append(dict(level='zone', cell=zn, share=base[k],
                         e_p5=np.percentile(e, 5), e_median=np.median(e),
                         e_p95=np.percentile(e, 95), pct_below_eps=100 * low))
        print(f'{zn:<17}{base[k]:>7.3f}{np.percentile(e, 5):>8.3f}'
              f'{np.median(e):>8.3f}{np.percentile(e, 95):>7.3f}{100*low:>10.1f}%')
    print('  a zone with most corners below the threshold has no support in')
    print('  most states: its effect can only be estimated where it IS chosen.')

    # ---- 2. positivity by state bins -------------------------------------
    print('\n=== 2. positivity by state bins (packed box x attack shape x swing) ===')
    sb = state_bins(d)
    tab = pd.crosstab(sb, d.lateral).reindex(columns=ZONES, fill_value=0)
    tab['n'] = tab.sum(axis=1)
    tab = tab.sort_values('n', ascending=False)
    print(tab.to_string())
    big = tab[tab.n >= 30]
    empty = (big[ZONES] < 5).sum()
    print(f'\n  among the {len(big)} states with n >= 30, cells with < 5 corners:')
    for zn in ZONES:
        print(f'    {zn:<16} {int(empty[zn])} of {len(big)} states')
    for s, r in tab.iterrows():
        for zn in ZONES:
            prow.append(dict(level='state', cell=f'{s} -> {zn}', share=r[zn] / r.n,
                             e_p5=np.nan, e_median=np.nan, e_p95=np.nan,
                             pct_below_eps=np.nan, n=int(r[zn]), n_state=int(r.n)))
    pd.DataFrame(prow).to_csv(f'{SP}/results/delivery_target_positivity.csv',
                              index=False)

    # ---- 3. contrasts ----------------------------------------------------
    print(f'\n=== 3. zone vs {REF}: naive vs adjusted '
          f'(trim {TRIM}, {N_BOOT} match-cluster bootstraps) ===')
    rows = []
    for outcome in OUTCOMES:
        print(f'\n  {outcome}')
        print(f'  {"zone":<17}{"n":>6}{"trim":>6}{"psAUC":>7}'
              f'{"naive":>9}{"IPW":>9}{"AIPW":>9}   AIPW 95% CI')
        for zn in ZONES:
            if zn == REF:
                continue
            r = contrast(d, zn, outcome)
            rows.append(r)
            sig = '' if r['aipw_lo'] < 0 < r['aipw_hi'] else '  *'
            print(f'  {zn:<17}{r["n_zone"]:>6}{r["trimmed_pct"]:>5.0f}%'
                  f'{r["ps_auc"]:>7.3f}{r["naive"]:>+9.4f}{r["ipw"]:>+9.4f}'
                  f'{r["aipw"]:>+9.4f}   [{r["aipw_lo"]:+.4f}, {r["aipw_hi"]:+.4f}]{sig}')
    res = pd.DataFrame(rows)
    res.to_csv(f'{SP}/results/delivery_target_causal.csv', index=False)

    x = res[res.outcome == 'xg_after'].set_index('zone')
    print('\n  Reading (xG, zone minus central):')
    for zn in x.index:
        r = x.loc[zn]
        print(f'    {zn:<16} naive {r.naive:+.4f} -> AIPW {r.aipw:+.4f} '
              f'(adjusting for the setup changes it by {r.aipw - r.naive:+.4f})')
    print('\n  Treatment is the ARRIVAL zone, not the aim: interceptions are part')
    print('  of it. Personnel is unobserved. Read as better-identified, not causal.')
    print('\nwrote results/delivery_target_causal.csv, '
          'results/delivery_target_positivity.csv')


if __name__ == '__main__':
    main()

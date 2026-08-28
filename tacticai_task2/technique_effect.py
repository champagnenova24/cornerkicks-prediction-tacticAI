"""Does the swing of the delivery change anything that matters?

This is the evidence the delivery recommendation in `tactic_recommend.py` rests
on, so it is worth being able to re-run rather than quoting from memory.

Technique is not a free choice: corner side crossed with the taker's foot fixes
the swing (2,153 long corners, no counterexamples). So "change the technique"
means "change the taker", which is a real personnel cost. The question is
whether the swing buys anything to justify that.

Three passes:
  1. raw contrast, outswinging vs inswinging, on attempts / xG / goals
  2. IPW adjustment on the defensive setup, cluster-bootstrapped over matches
  3. an interaction scan -- if the average is null but the effect is real in
     some state, a conditional recommendation would beat a constant one

Outputs: technique_effect.csv, technique_interactions.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import outdirs  # noqa: F401 -- creates results/ figures/ models/
SP = os.path.dirname(os.path.abspath(__file__))
MIN_CHORD = 15.0
TRIM = (0.05, 0.95)
N_BOOT = 600
MIN_STRATUM = 80

COV = ['def_in6', 'def_in6_frac', 'mark_dist', 'marked_frac', 'def_line_x',
       'def_deepest_x', 'def_spread_y', 'near_post_cover', 'far_post_cover',
       'gk_depth', 'gk_y_off', 'n_def_visible']
OUTCOMES = ['shot', 'xg_after', 'goal_after']


def load():
    d = pd.read_parquet(f'{SP}/defense_features.parquet')
    d = d[(d.chord >= MIN_CHORD)
          & d.technique.isin(['Inswinging', 'Outswinging'])].copy()
    d['out'] = (d.technique == 'Outswinging').astype(int)
    return d


def propensity(d):
    X = np.nan_to_num(d[COV].to_numpy(float))
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return LogisticRegression(max_iter=2000).fit(X, d.out.to_numpy()) \
        .predict_proba(X)[:, 1]


def ipw(t, y, ps):
    sel = (ps > TRIM[0]) & (ps < TRIM[1])
    t, y, ps = t[sel], y[sel], ps[sel]
    if len(np.unique(t)) < 2:
        return np.nan
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    return ((w * y * t).sum() / (w * t).sum()
            - (w * y * (1 - t)).sum() / (w * (1 - t)).sum())


def cluster_ci(d, y, ps, stat, n_boot=N_BOOT, seed=0):
    """Resample whole matches: corners in a match share taker and tactics."""
    mids = d.match_id.to_numpy()
    um = np.unique(mids)
    where = {m: np.where(mids == m)[0] for m in um}
    rng = np.random.default_rng(seed)
    t = d.out.to_numpy()
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([where[m] for m in
                              rng.choice(um, len(um), replace=True)])
        if len(np.unique(t[idx])) < 2:
            continue
        v = stat(t[idx], y[idx], ps[idx])
        if np.isfinite(v):
            bs.append(v)
    return (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan,) * 2


def main():
    d = load()
    ps = propensity(d)
    print(f'{len(d):,} long corners  ({int(d.out.sum()):,} outswinging, '
          f'{int((1 - d.out).sum()):,} inswinging)')
    print(f'propensity AUC on the defensive setup: '
          f'{__import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(d.out, ps):.3f}')

    rows = []
    print(f'\n=== outswinging - inswinging ===')
    print(f'{"outcome":<12}{"raw":>10}{"IPW":>10}{"  95% CI (cluster)":>24}')
    for o in OUTCOMES:
        y = d[o].to_numpy(float)
        raw = y[d.out == 1].mean() - y[d.out == 0].mean()
        adj = ipw(d.out.to_numpy(), y, ps)
        lo, hi = cluster_ci(d, y, ps, ipw)
        crosses = lo < 0 < hi
        print(f'{o:<12}{raw:>+10.4f}{adj:>+10.4f}   [{lo:+.4f},{hi:+.4f}]'
              f'{"  crosses 0" if crosses else "  *"}')
        rows.append(dict(outcome=o, raw=raw, ipw=adj, ci_lo=lo, ci_hi=hi,
                         significant=not crosses))
    pd.DataFrame(rows).to_csv(f'{SP}/results/technique_effect.csv', index=False)

    # ---- interaction scan -------------------------------------------------
    strata = {}
    for name, col in [('packed 6-yd box', 'def_in6'),
                      ('zonal marking', 'mark_dist'),
                      ('GK off line', 'gk_depth')]:
        med = d[col].median()
        strata[f'{name}: high'] = d[d[col] >= med]
        strata[f'{name}: low'] = d[d[col] < med]
    for lab in ['near post', 'central', 'far post']:
        strata[f'target: {lab}'] = d[d.lateral == lab]

    print(f'\n=== interaction scan: is the null hiding a real effect? ===')
    print(f'{"stratum":<26}{"n_out":>6}{"n_in":>6}{"dXG":>9}'
          f'{"  95% CI":>21}{"dSHOT":>9}')
    irows = []
    for k, s in strata.items():
        if len(s) < MIN_STRATUM or s.out.nunique() < 2:
            continue
        dxg = s[s.out == 1].xg_after.mean() - s[s.out == 0].xg_after.mean()
        dsh = s[s.out == 1].shot.mean() - s[s.out == 0].shot.mean()
        lo, hi = cluster_ci(
            s, s.xg_after.to_numpy(float), np.zeros(len(s)),
            lambda t, y, _: y[t == 1].mean() - y[t == 0].mean())
        sig = not (lo < 0 < hi)
        print(f'{k:<26}{int(s.out.sum()):>6}{int((1 - s.out).sum()):>6}'
              f'{dxg:>+9.4f}  [{lo:+.4f},{hi:+.4f}]{dsh:>+9.4f}'
              f'{"  *" if sig else ""}')
        irows.append(dict(stratum=k, n_out=int(s.out.sum()),
                          n_in=int((1 - s.out).sum()), d_xg=dxg, d_shot=dsh,
                          ci_lo=lo, ci_hi=hi, significant=sig))
    idf = pd.DataFrame(irows)
    idf.to_csv(f'{SP}/results/technique_interactions.csv', index=False)

    n_sig = int(idf.significant.sum())
    print(f'\n  {n_sig} of {len(idf)} strata show an xG effect '
          f'(expect ~{0.05 * len(idf):.1f} by chance at 5%)')
    print(f'  shot rate favours outswinging in '
          f'{int((idf.d_shot > 0).sum())} of {len(idf)}')
    print('\n  Reading: the swing buys attempts in essentially every state and')
    print('  value in none. Since switching swing means switching takers, and')
    print('  the freeze frame cannot see who strikes a ball better, the lever')
    print('  to pull is the target zone -- not the technique.')
    print('\n  Null is not equivalence: the CI above still admits an effect of')
    print('  roughly a third of the base xG rate in either direction.')
    print('\nwrote results/technique_effect.csv, results/technique_interactions.csv')


if __name__ == '__main__':
    main()

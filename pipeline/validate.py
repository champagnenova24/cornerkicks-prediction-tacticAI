"""Sanity checks over the processed parquets. Run after stages 1-5."""
import pandas as pd

from common import PROCESSED, DT_CLIP


def main():
    m = pd.read_parquet(f'{PROCESSED}/matches.parquet')
    p = pd.read_parquet(f'{PROCESSED}/possessions.parquet')
    d = pd.read_parquet(f'{PROCESSED}/decisions.parquet',
                        columns=['match_id', 'type', 'zone', 'dt', 'x', 'y',
                                 'end_x', 'end_y', 'xg_after', 'goal_after',
                                 'success', 'off_camera', 'n_remaining'])
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet')

    print('=== corpus ===')
    print(f'matches {len(m):,} | possessions {len(p):,} | decisions {len(d):,} '
          f'| 360 decision points {len(dp):,}')

    print('\n=== possessions ===')
    print(p['terminal_outcome'].value_counts().to_string())
    print(f"goal rate: {p['goal'].mean()*100:.2f}%  "
          f"mean decisions/possession: {p['n_decisions'].mean():.2f}")
    print('decisions/possession quantiles:',
          p['n_decisions'].quantile([.5, .9, .95, .99]).to_dict())

    print('\n=== decisions ===')
    print(d['type'].value_counts().to_string())
    assert d[['x', 'y']].min().min() >= 0 and d['x'].max() <= 121 and d['y'].max() <= 81
    assert d['dt'].between(0, DT_CLIP).all()
    assert (d.loc[d['type'] == 'shot', 'zone'] == -1).all()
    print(f"dt clipped share: {(d['dt'] == DT_CLIP).mean()*100:.2f}%")
    print(f"off_camera share: {d['off_camera'].mean()*100:.2f}%")
    print(f"mean xg_after: {d['xg_after'].mean():.4f} | goal_after rate: "
          f"{d['goal_after'].mean()*100:.2f}%")

    print('\n=== 360 decision points ===')
    print(f"visible players/frame: mean {dp['n_visible'].mean():.1f}, "
          f"min {dp['n_visible'].min()}, max {dp['n_visible'].max()}")
    print(f"opponent keeper visible: {dp['keeper_visible'].mean()*100:.1f}%")
    print(f"actor in frame: {dp['actor_in_frame'].mean()*100:.1f}%")
    print(dp['action_type'].value_counts().to_string())

    print('\n=== splits (decisions by match split) ===')
    dm = d.merge(m[['match_id', 'corpus', 'split']], on='match_id')
    print(dm.groupby(['corpus', 'split']).size().unstack(fill_value=0).to_string())

    dpm = dp.merge(m[['match_id', 'split']], on='match_id')
    print('\n360 decision points by split:')
    print(dpm['split'].value_counts().to_string())

    # positivity preview: action mix by ball-location third
    print('\n=== positivity preview (action share by pitch third) ===')
    dp['third'] = pd.cut(dp['x'], [0, 40, 80, 121], labels=['def', 'mid', 'att'])
    print(pd.crosstab(dp['third'], dp['action_type'], normalize='index')
          .round(3).to_string())


if __name__ == '__main__':
    main()

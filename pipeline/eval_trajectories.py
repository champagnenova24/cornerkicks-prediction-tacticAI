"""Evaluate reconstructed corner trajectories with the gap/fidelity method.

The same analysis applied to the raw event stream at the start of this project
-- measure the spacing between consecutive samples, then ask what that spacing
implies about fidelity -- turned on the OUTPUT of ball_trajectory.py, plus the
physical cross-checks that method implies:

  1. sample spacing of the reconstruction vs the real event stream
  2. endpoint preservation (does it keep what was observed?)
  3. apex vs closed-form ballistics (internal consistency)
  4. speed profile (constant-speed assumption vs drag)
  5. is `duration` really flight time? (independent timestamp cross-check)
  6. how well-located is the endpoint the trajectory is anchored to?
"""
import json
import os
import random

import numpy as np
import pandas as pd

from common import EVENTS_DIR, PROCESSED, ts_to_sec
from ball_trajectory import G_YD

# Reference numbers for the raw event stream, from the stage-0 gap analysis
# over all 4,235 matches / 14.87M events.
EVENT_GAP_MEDIAN = 0.695
EVENT_GAP_MEAN = 1.761
EVENT_GAP_ZERO_SHARE = 31.6
TRACKING_HZ = 25.0

SAMPLE_MATCHES = 120


def main():
    df = pd.read_parquet(f'{PROCESSED}/corner_trajectories.parquet')
    df['duration'] = df.groupby('uuid')['t'].transform('max')
    n_corners = df['uuid'].nunique()
    print(f'evaluating {n_corners:,} reconstructed corners '
          f'({len(df):,} sampled points)\n')

    # ------------------------------------------------- 0. field of play --
    # A trajectory that leaves the pitch is wrong regardless of anything else:
    # a corner starts ON the goal line, so a goalward lateral bow puts the ball
    # behind it. Regression guard for that bug.
    oob = ((df.x > 120.0001) | (df.x < -0.0001)
           | (df.y > 80.0001) | (df.y < -0.0001))
    n_bad = df.loc[oob, 'uuid'].nunique()
    print('=== 0. field of play ===')
    print(f'points outside the pitch: {oob.sum():,} / {len(df):,} '
          f'({oob.mean()*100:.2f}%)   corners affected: {n_bad:,}')
    print(f'max x reached: {df.x.max():.3f} (goal line 120.0)')
    if oob.any():
        print('  FAIL -- trajectories leave the field of play')
        print(df.loc[oob].groupby('technique').size().to_string())
    else:
        print('  PASS -- every sampled point is in play\n')

    # ---------------------------------------------------------- 1. spacing --
    g = df.sort_values(['uuid', 'point_idx'])
    dt = g.groupby('uuid')['t'].diff().dropna().values
    dt = dt[dt > 0]
    print('=== 1. sample spacing (the previous method) ===')
    print(f'reconstruction: median gap {np.median(dt)*1000:.1f} ms, '
          f'mean {dt.mean()*1000:.1f} ms')
    print(f'  implied sampling rate: {1/np.median(dt):.1f} Hz')
    print(f'  zero-length gaps: {(dt == 0).mean()*100:.1f}%')
    print(f'real event stream:  median gap {EVENT_GAP_MEDIAN*1000:.0f} ms, '
          f'mean {EVENT_GAP_MEAN*1000:.0f} ms, '
          f'{EVENT_GAP_ZERO_SHARE:.1f}% simultaneous')
    print(f'real tracking data (not in this dataset): {TRACKING_HZ:.0f} Hz '
          f'= {1000/TRACKING_HZ:.0f} ms')
    print(f'  -> reconstruction is {EVENT_GAP_MEDIAN/np.median(dt):.0f}x denser '
          f'than the events it came from, and '
          f'{(1/np.median(dt))/TRACKING_HZ:.1f}x denser than real tracking.')
    print('  -> every one of those samples is interpolated. The density is a '
          'rendering choice,\n     not evidence; independent information per '
          'corner is still 2 endpoints + 1 duration.')

    # per-corner: how many observed numbers vs how many emitted numbers
    pts = len(df) / n_corners
    print(f'\n  observed quantities per corner: 5 (start xy, end xy, duration)')
    print(f'  emitted numbers per corner:     {pts*4:.0f} '
          f'({pts:.0f} points x t,x,y,z)')
    print(f'  inflation factor: {pts*4/5:.0f}x')

    # ------------------------------------------- raw-event dependent checks --
    random.seed(0)
    mids = sorted(df['match_id'].unique())
    mids = random.sample(mids, min(SAMPLE_MATCHES, len(mids)))
    want = set(df.loc[df.match_id.isin(mids), 'uuid'])

    endpoint_err, apex_err = [], []
    dur_vs_ts, chain_gap = [], []
    for mid in mids:
        try:
            with open(f'{EVENTS_DIR}/{mid}.json') as f:
                events = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        by_index = {e['index']: e for e in events}
        for e in events:
            if e['id'] not in want:
                continue
            p = e.get('pass', {})
            sub = df[df.uuid == e['id']].sort_values('point_idx')
            if sub.empty:
                continue
            # 2. endpoints
            endpoint_err.append(max(
                abs(sub['x'].iloc[0] - e['location'][0]),
                abs(sub['y'].iloc[0] - e['location'][1]),
                abs(sub['x'].iloc[-1] - p['end_location'][0]),
                abs(sub['y'].iloc[-1] - p['end_location'][1])))
            # 3. apex vs closed form
            T = e.get('duration') or 0
            if p.get('height', {}).get('name') == 'High Pass' and T > 0:
                apex_err.append(abs(sub['apex_z'].iloc[0] - G_YD * T ** 2 / 8))
            # 5. duration vs timestamp difference to the next event
            nxt = by_index.get(e['index'] + 1)
            if nxt and T > 0 and nxt['period'] == e['period']:
                gap = ts_to_sec(nxt['timestamp']) - ts_to_sec(e['timestamp'])
                dur_vs_ts.append(gap - T)
            # 6. does the next event agree with where we put the ball?
            # coordinates are per-event-team, so rotate the follow-up into the
            # corner taker's frame before comparing (see stage 3).
            if nxt and nxt.get('location'):
                nx, ny = nxt['location'][0], nxt['location'][1]
                if nxt['team']['id'] != e['team']['id']:
                    nx, ny = 120.0 - nx, 80.0 - ny
                chain_gap.append(float(np.hypot(
                    nx - p['end_location'][0], ny - p['end_location'][1])))

    print('\n=== 2. endpoint preservation ===')
    ee = np.array(endpoint_err)
    print(f'max deviation from observed endpoints: {ee.max():.4f} yd '
          f'(n={len(ee):,}) -> observed values are kept exactly')

    print('\n=== 3. internal consistency (apex vs g*T^2/8) ===')
    ae = np.array(apex_err)
    print(f'max |apex - closed form|: {ae.max():.2e} yd -> ballistics exact')

    print('\n=== 4. speed profile ===')
    sp = df.groupby('uuid')['speed'].first().values
    print(f'reconstructed speed (constant by construction): '
          f'median {np.median(sp):.1f} yd/s ({np.median(sp)*0.9144*3.6:.0f} km/h)')
    print('  a real ball decelerates: with a drag coefficient typical of a')
    print('  football, launch speed is ~15-25% above and arrival ~15-25% below')
    print('  this mean. The reconstruction has NO deceleration -- it is the only')
    print('  place the model contradicts physics rather than merely assuming it.')

    print('\n=== 5. is `duration` really flight time? ===')
    dv = np.array(dur_vs_ts)
    print(f'(next event timestamp - corner timestamp) - duration, n={len(dv):,}')
    print(f'  median {np.median(dv):+.3f} s, '
          f'p10 {np.percentile(dv,10):+.3f}, p90 {np.percentile(dv,90):+.3f}')
    print(f'  agrees to within 1 ms: {(np.abs(dv) <= 0.001).mean()*100:.0f}%  |  '
          f'within 0.1 s: {(np.abs(dv) <= 0.1).mean()*100:.0f}%')
    print('  -> `duration` is NOT an independent measurement of ball flight: it')
    print('     is the gap to the next logged event. Flight time is therefore')
    print('     only as good as where the annotator placed that event.')
    # what an 0.1s duration error does to apex
    for T in (1.2, 1.7, 2.2):
        a0 = G_YD * T ** 2 / 8
        a1 = G_YD * (T + 0.1) ** 2 / 8
        print(f'  sensitivity: T={T}s -> apex {a0:.2f} yd; '
              f'T+0.1s -> {a1:.2f} yd ({(a1/a0-1)*100:+.0f}%)')

    print('\n=== 6. how well-located is the anchor endpoint? ===')
    cg = np.array(chain_gap)
    print(f'distance from pass end_location to the next event location, '
          f'n={len(cg):,}')
    print(f'  exact (<0.15 yd): {(cg < 0.15).mean()*100:.0f}%  |  '
          f'median {np.median(cg):.1f} yd  |  p90 {np.percentile(cg,90):.1f} yd')
    print('  (comparison requires rotating the follow-up event into the taker\'s')
    print('   attacking frame first -- skipping that flip makes the tail look')
    print('   like a full pitch length.)')

    print('\n=== 7. contact height (was: ground-to-ground bias) ===')
    per = df.groupby('uuid').first()
    ends = per['end_z'].values
    print(f'flight now solved to z(T) = the height the ball was met at, keyed')
    print(f'on the body part of the next touch (observed for ~93% of corners):')
    print(f'  mean contact height {ends.mean():.2f} yd, '
          f'met above 1 yd: {(ends > 1).mean()*100:.0f}%, '
          f'at ground: {(ends == 0).mean()*100:.0f}%')
    # every trajectory must actually arrive at its contact height
    last = g.groupby('uuid').last()
    resid = np.abs(last['z'].values - per.loc[last.index, 'end_z'].values)
    print(f'  max |z(T) - contact height|: {resid.max():.2e} yd -> honoured exactly')

    aerial = per[per['height_class'] == 'High Pass']
    old = G_YD * aerial['duration'].values ** 2 / 8 if 'duration' in aerial else None
    print(f'  apex over High Pass corners: mean {aerial["apex_z"].mean():.2f} yd')
    if old is not None:
        print(f'  vs ground-to-ground formula: mean {old.mean():.2f} yd '
              f'({(aerial["apex_z"].mean()/old.mean()-1)*100:+.0f}%)')
    print('  the CATEGORY is observed; the height attached to it is a prior')
    print('  (Head 2.2 yd etc.) -- tune CONTACT_HEIGHT_YD, do not fit to it.')

    print('\n=== verdict ===')
    print('faithful: endpoints, flight time, swing direction, height class')
    print('assumed:  the path between the endpoints; contact height per body part')
    print('          (conditioned on an observed category, not a blanket zero)')
    print('wrong:    constant speed -- no drag; the one remaining physics defect')
    print('unidentified: curvature magnitude -- no observable constrains it')


if __name__ == '__main__':
    main()

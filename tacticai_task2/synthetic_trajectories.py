"""Ball trajectories for SYNTHETIC corners, and a ball-consistency check.

The diffusion model generates player positions and velocities -- it does not
generate a delivery. Every synthetic corner is sampled on a real corner's
roster template, so the delivery that belongs to that template is the ball
path for the generated setup: the model is answering "where would the players
stand for THIS corner", with the corner itself held fixed.

That makes the ball more than decoration. It supplies a test the project has
not had: a real corner is a configuration of players organised AROUND a
delivery, so if the generator has learned that structure, its players should
converge on the arrival point the way the real ones do. If they ignore it, the
setups are plausible-looking crowds rather than responses to a specific ball.

Two distances per corner, from the ball's arrival point:
  at kick    -- nearest player in the freeze frame, when the corner is struck
  at arrival -- nearest player after moving each one by v * flight_time

The second is the meaningful one. Velocities here are ball-flight AVERAGES
(displacement over exactly the flight, see README), so x + v*T lands a player
approximately where they were when the ball got there -- the estimate and the
extrapolation cover the same interval by construction.

Physics is imported from the pipeline's ball_trajectory module rather than
reimplemented; it is the single source of truth for the reconstruction, and
the assumptions documented there (swing magnitude is a prior, contact height
is a per-category prior) carry over unchanged.

Outputs: synthetic_corner_trajectories.parquet, ball_consistency.csv,
         synthetic_trajectories.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(os.path.dirname(SP), 'pipeline')
sys.path.insert(0, PIPELINE)

from ball_trajectory import reconstruct, end_height_from_followups  # noqa: E402
from common import EVENTS_DIR  # noqa: E402

N_POINTS = 60
VARIANTS = ['real', 'real_label', 'shot1', 'shot0']
NEAR_YD = 3.0


def reflect_xy(x, y, do):
    """tacticai frame: corners are mirrored so the kick is taken at (120, 0)."""
    return (x, 80.0 - y) if do else (x, y)


def deliveries(corners):
    """Delivery parameters per corner, read from the raw StatsBomb events.

    Returns one row per corner: reflected endpoints, flight time, height class,
    technique, and the height the ball was met at (from the body part of the
    following touch).
    """
    want = corners.groupby('match_id')['corner_id'].apply(set).to_dict()
    refl = corners.set_index('corner_id')['reflected'].to_dict()
    rows = []
    for k, (mid, uuids) in enumerate(want.items()):
        try:
            with open(f'{EVENTS_DIR}/{mid}.json') as f:
                events = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        events.sort(key=lambda e: e['index'])
        for i, e in enumerate(events):
            if e['id'] not in uuids or e['type']['name'] != 'Pass':
                continue
            p = e.get('pass', {})
            if not e.get('location') or not p.get('end_location'):
                continue
            do = bool(refl[e['id']])
            sx, sy = reflect_xy(float(e['location'][0]),
                                float(e['location'][1]), do)
            ex, ey = reflect_xy(float(p['end_location'][0]),
                                float(p['end_location'][1]), do)
            rows.append({
                'corner_id': e['id'], 'match_id': mid,
                'start_x': sx, 'start_y': sy, 'end_x': ex, 'end_y': ey,
                'duration': e.get('duration') or np.nan,
                'height_class': p.get('height', {}).get('name', 'High Pass'),
                'technique': p.get('technique', {}).get('name'),
                'end_height': end_height_from_followups(events[i + 1:i + 5]),
            })
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(want)} matches read', file=sys.stderr,
                  flush=True)
    return pd.DataFrame(rows)


def build_trajectories(dl):
    """reconstruct() per delivery -> long frame of sampled points."""
    out = []
    for r in dl.itertuples():
        if not np.isfinite(r.duration) or r.duration <= 0:
            continue
        tr = reconstruct(
            (r.start_x, r.start_y), (r.end_x, r.end_y), r.duration,
            height_class=r.height_class, technique=r.technique,
            n_points=N_POINTS, end_height=r.end_height)
        out.append(pd.DataFrame({
            'corner_id': r.corner_id, 'point_idx': np.arange(N_POINTS),
            't': tr['t'], 'x': tr['x'], 'y': tr['y'], 'z': tr['z'],
            'apex_z': tr['apex_z'], 'end_z': tr['end_z'], 'speed': tr['speed'],
            'height_class': tr['height_class'], 'technique': tr['technique'],
            'swing_used': tr['swing_used'], 'duration': tr['duration'],
        }))
    return pd.concat(out, ignore_index=True)


def consistency(players, dl, label):
    """Distance from the ball's arrival point to the nearest player.

    Measured twice: in the freeze frame (at the kick) and after advancing each
    player by v * flight_time (roughly where they are when the ball lands).
    """
    d = dl.set_index('corner_id')
    rows = []
    for cid, g in players.groupby('corner_id'):
        base = cid.split('__')[0]
        if base not in d.index:
            continue
        r = d.loc[base]
        T = float(r.duration)
        if not np.isfinite(T) or T <= 0:
            continue
        bx, by = float(r.end_x), float(r.end_y)
        x, y = g.x.to_numpy(float), g.y.to_numpy(float)
        vx, vy = g.vx.to_numpy(float), g.vy.to_numpy(float)
        att = g.teammate.to_numpy(bool) & ~g.actor.to_numpy(bool)
        d_kick = np.hypot(x - bx, y - by)
        d_arr = np.hypot(x + vx * T - bx, y + vy * T - by)
        if not att.any():
            continue
        rows.append({
            'corner_id': cid, 'variant': label,
            'd_any_kick': d_kick.min(), 'd_any_arrival': d_arr.min(),
            'd_att_kick': d_kick[att].min(),
            'd_att_arrival': d_arr[att].min(),
            'att_near_kick': float((d_kick[att] < NEAR_YD).any()),
            'att_near_arrival': float((d_arr[att] < NEAR_YD).any()),
            'n_att_near_arrival': float((d_arr[att] < NEAR_YD).sum()),
        })
    return pd.DataFrame(rows)


def draw(ax, g, traj, title):
    ax.add_patch(plt.Rectangle((90, 0), 30, 80, fill=False, lw=1, ec='gray'))
    ax.plot([102, 102], [18, 62], c='gray', lw=1)
    ax.plot([102, 120], [18, 18], c='gray', lw=1)
    ax.plot([102, 120], [62, 62], c='gray', lw=1)
    ax.add_patch(plt.Rectangle((114, 30), 6, 20, fill=False, lw=1, ec='gray'))
    ax.plot([120, 120], [36, 44], c='black', lw=3)
    if traj is not None:
        z = traj.z.to_numpy()
        ax.plot(traj.x, traj.y, c='darkorange', lw=1.2, alpha=0.5, zorder=2)
        ax.scatter(traj.x, traj.y, s=2 + 9 * z / max(z.max(), 1e-6),
                   c='darkorange', zorder=2, alpha=0.8)
        ax.scatter([traj.x.iloc[-1]], [traj.y.iloc[-1]], marker='X', s=90,
                   c='darkorange', edgecolors='black', linewidths=0.6, zorder=6)
    for r in g.itertuples():
        c = 'tab:red' if r.teammate else 'tab:blue'
        mk = '*' if r.actor else ('s' if r.keeper else 'o')
        ax.scatter(r.x, r.y, c=c, marker=mk, s=70 if r.actor else 35,
                   zorder=4, edgecolors='black', linewidths=0.4)
        if r.has_vel and (abs(r.vx) + abs(r.vy)) > 0.3:
            ax.arrow(r.x, r.y, r.vx, r.vy, head_width=0.8, color=c, alpha=0.6)
    ax.set_xlim(88, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)


def main():
    corners = pd.read_parquet(f'{SP}/corners.parquet')
    real = pd.read_parquet(f'{SP}/corner_players.parquet')
    syn = pd.read_parquet(f'{SP}/synthetic_corners.parquet')
    split = pd.read_csv(f'{SP}/diffusion_split.csv')
    hold = set(split[split.split == 'holdout'].corner_id)

    # Reading 416 event files is the slow part and the result never changes;
    # cache it so re-runs (a new figure, a tweaked metric) are instant.
    cache = f'{SP}/corner_deliveries.parquet'
    if os.path.exists(cache):
        dl = pd.read_parquet(cache)
        print(f'deliveries: {len(dl):,} (cached, delete {os.path.basename(cache)} '
              f'to rebuild)', flush=True)
    else:
        print(f'reading deliveries for {len(corners):,} corners '
              f'across {corners.match_id.nunique()} matches', flush=True)
        dl = deliveries(corners)
        dl.to_parquet(cache, index=False)
        print(f'deliveries recovered: {len(dl):,}', flush=True)

    traj = build_trajectories(dl)
    traj.to_parquet(f'{SP}/synthetic_corner_trajectories.parquet', index=False)
    per = traj.groupby('corner_id').first()
    print(f'\nwrote synthetic_corner_trajectories.parquet: '
          f'{traj.corner_id.nunique():,} corners, {len(traj):,} points')
    print(f'  apex height    mean {per.apex_z.mean():.2f} yd')
    print(f'  contact height mean {per.end_z.mean():.2f} yd, '
          f'met above 1 yd: {(per.end_z > 1).mean()*100:.0f}%')
    swung = per[per.technique.isin(['Inswinging', 'Outswinging'])]
    print(f'  swing clamped by the touchline on '
          f'{(swung.swing_used < 2.49).mean()*100:.0f}% of the '
          f'{len(swung):,} swung deliveries')

    # ---- ball consistency, holdout templates only ----
    dl_h = dl[dl.corner_id.isin(hold)]
    real_h = real[real.corner_id.isin(hold)]
    parts = [consistency(real_h, dl_h, 'real')]
    for v in ['real_label', 'shot1', 'shot0']:
        parts.append(consistency(syn[syn.variant == v], dl_h, v))
    cons = pd.concat(parts, ignore_index=True)
    cons.to_csv(f'{SP}/ball_consistency.csv', index=False)

    print(f'\n=== ball consistency on {cons[cons.variant=="real"].shape[0]} '
          f'holdout corners ===')
    print('distance (yd) from the ball arrival point to the nearest attacker\n')
    agg = cons.groupby('variant').agg(
        n=('corner_id', 'size'),
        att_kick_median=('d_att_kick', 'median'),
        pct_within_3yd_kick=('att_near_kick', lambda s: 100 * s.mean()),
        att_arrival_median=('d_att_arrival', 'median'),
        att_arrival_mean=('d_att_arrival', 'mean'),
        any_arrival_median=('d_any_arrival', 'median'),
        pct_att_within_3yd=('att_near_arrival', lambda s: 100 * s.mean()),
        n_att_within_3yd=('n_att_near_arrival', 'mean'),
    ).reindex(VARIANTS).round(2)
    print(agg.to_string())
    print('\n  "kick" columns use freeze-frame positions only. "arrival"')
    print('  columns advance every player by v * flight_time. Real velocities')
    print('  are estimated from where players actually ended up, so the real')
    print('  arrival figure is partly definitional -- the question is whether')
    print('  the generator reproduces that structure, not whether it beats it.')

    r = agg.loc['real']
    s = agg.loc['real_label']
    print(f'\n  real setups put an attacker within {NEAR_YD:.0f} yd of the '
          f'arrival point on {r.pct_att_within_3yd:.0f}% of corners;')
    print(f'  generated setups on the same deliveries: '
          f'{s.pct_att_within_3yd:.0f}%.')
    print(f'  median nearest-attacker distance at arrival: '
          f'{r.att_arrival_median:.2f} yd real vs '
          f'{s.att_arrival_median:.2f} yd synthetic '
          f'({s.att_arrival_median - r.att_arrival_median:+.2f}).')

    # ---- figure ----
    # Show the modal case: a high delivery actually crossed into the box.
    # Sorting holdout ids and taking the first three lands on short corners,
    # which are 13% of the data and not what the metric is about.
    chord = np.hypot(dl.end_x - dl.start_x, dl.end_y - dl.start_y)
    pick = dl[(dl.height_class == 'High Pass') & (chord > 15)
              & dl.corner_id.isin(hold) & dl.corner_id.isin(set(traj.corner_id))]
    ids = pick.sort_values('corner_id').corner_id.tolist()[:3]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for col, cid in enumerate(ids):
        t = traj[traj.corner_id == cid]
        draw(axes[0, col], real[real.corner_id == cid], t, 'real')
        draw(axes[1, col], syn[(syn.variant == 'real_label')
                               & (syn.corner_id == f'{cid}__real_label')],
             t, 'synthetic (same delivery)')
    fig.suptitle('Reconstructed delivery over real vs generated setups\n'
                 'orange = ball path (dot size = height), X = arrival point',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{SP}/synthetic_trajectories.png', dpi=150)
    print('\nwrote ball_consistency.csv, synthetic_trajectories.png')


if __name__ == '__main__':
    main()

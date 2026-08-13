"""Render reconstructed corner-kick ball trajectories over 360 freeze frames.

These are REAL corners from the dataset: player positions and the ball's
endpoints/flight time are observed, the path between them is reconstructed by
ball_trajectory.py. No generative model is involved.

Usage:  python viz_corners.py [outdir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

from ball_trajectory import reconstruct_event
from common import EVENTS_DIR, THREESIXTY_DIR

# Categorical slots 1-3 of the reference palette (validated for all-pairs use).
ATTACK = '#2a78d6'   # slot 1 blue
DEFEND = '#eb6834'   # slot 2 orange
BALL = '#1baf7a'     # slot 3 aqua
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
LINE = '#d8d7d2'     # recessive pitch markings


def draw_pitch(ax):
    """Attacking third drawn landscape: pitch (x, y) is plotted as (y, x), so
    the goal being attacked sits along the top edge. Markings stay recessive."""
    lw = 1.0
    ax.add_patch(plt.Rectangle((0, 60), 80, 60, fill=False, ec=LINE, lw=lw, zorder=1))
    # penalty area, six-yard box, goal
    ax.add_patch(plt.Rectangle((18, 102), 44, 18, fill=False, ec=LINE, lw=lw, zorder=1))
    ax.add_patch(plt.Rectangle((30, 114), 20, 6, fill=False, ec=LINE, lw=lw, zorder=1))
    ax.add_patch(plt.Rectangle((36, 120), 8, 1.6, fill=False, ec=INK_2, lw=1.4, zorder=1))
    ax.add_patch(Circle((40, 108), 0.35, color=LINE, zorder=1))
    # penalty arc, bulging away from goal
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, ec=LINE, lw=lw, zorder=1))
    # corner arcs, opening into the pitch
    ax.add_patch(Arc((0, 120), 4, 4, theta1=270, theta2=360, ec=LINE, lw=lw, zorder=1))
    ax.add_patch(Arc((80, 120), 4, 4, theta1=180, theta2=270, ec=LINE, lw=lw, zorder=1))
    ax.set_xlim(-2, 82)
    ax.set_ylim(84, 123)   # final ~36 yd; corners and their targets all sit here
    ax.set_aspect('equal')
    ax.axis('off')


def panel(ax_top, ax_side, match_id, uuid, title, note):
    with open(f'{EVENTS_DIR}/{match_id}.json') as f:
        events = sorted(json.load(f), key=lambda e: e['index'])
    idx = next(i for i, e in enumerate(events) if e['id'] == uuid)
    event = events[idx]
    with open(f'{THREESIXTY_DIR}/{match_id}.json') as f:
        frame = next(fr for fr in json.load(f) if fr['event_uuid'] == uuid)

    # follow-ups identify the body part that met the delivery, and so the
    # height the flight is solved to arrive at
    traj = reconstruct_event(event, followups=events[idx + 1:idx + 5])
    ff = frame['freeze_frame']

    draw_pitch(ax_top)

    # players: 2px surface ring keeps overlapping dots separable.
    # pitch (x, y) plots as (y, x) so the goal sits along the top edge.
    for p in ff:
        if p.get('actor'):
            continue
        px, py = p['location']
        ax_top.plot(py, px, 'o', ms=8, color=ATTACK if p['teammate'] else DEFEND,
                    mec=SURFACE, mew=2, zorder=3)

    # ball path: width grows with height so the top-down view still shows z
    x, y, z = traj['x'], traj['y'], traj['z']
    zmax = max(z.max(), 1e-6)
    for i in range(len(x) - 1):
        ax_top.plot(y[i:i + 2], x[i:i + 2], '-', color=BALL,
                    lw=1.4 + 2.6 * (z[i] / zmax), solid_capstyle='round', zorder=4)
    ax_top.plot(y[0], x[0], 'o', ms=9, color=BALL, mec=SURFACE, mew=2, zorder=5)
    ax_top.plot(y[-1], x[-1], 'X', ms=11, color=BALL, mec=SURFACE, mew=1.6, zorder=5)

    ax_top.set_title(f'{title}\n{note}', fontsize=10.5, color=INK, loc='left', pad=6)

    # side profile: height against distance travelled
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    ax_side.fill_between(dist, 0, z, color=BALL, alpha=0.16, lw=0)
    ax_side.plot(dist, z, '-', color=BALL, lw=2, solid_capstyle='round')
    ax_side.axhline(0, color=LINE, lw=1)
    # mark where the ball was met, when that is off the ground
    if traj['end_z'] > 0.05:
        ax_side.plot([dist[-1]], [traj['end_z']], 'o', ms=7, color=BALL,
                     mec=SURFACE, mew=1.6, zorder=4)
    ax_side.set_ylim(0, 7.6)
    ax_side.set_xlim(0, max(dist.max(), 1))
    ax_side.spines[['top', 'right', 'left']].set_visible(False)
    ax_side.spines['bottom'].set_color(LINE)
    ax_side.tick_params(labelsize=7.5, colors=INK_2, length=0)
    ax_side.set_yticks([0, 3, 6])
    ax_side.set_xlim(left=0)
    ax_side.grid(axis='y', color=LINE, lw=0.6, alpha=0.7)
    ax_side.set_axisbelow(True)
    ax_side.set_xlabel('distance along path (yd)', fontsize=7.5, color=INK_2, labelpad=2)
    ax_side.set_ylabel('height (yd)', fontsize=7.5, color=INK_2, labelpad=2)
    met = (f" · met at {traj['end_z']:.1f} yd" if traj['end_z'] > 0.05
           else ' · reached the ground')
    ax_side.text(0.99, 0.88,
                 f"apex {traj['apex_z']:.1f} yd · flight {traj['duration']:.2f}s{met}",
                 transform=ax_side.transAxes, fontsize=8, color=INK_2,
                 ha='right', va='top')
    return traj


def main(outdir='.'):
    corners = [
        (3773661, 'f594be0e-51ba-4f76-989b-c8e550d0282c',
         'Inswinging — assisted a goal', 'High pass · complete'),
        (3773387, '7e741f7c-9b71-4249-a3af-fc1dfcb67837',
         'Outswinging — assisted a shot', 'High pass · complete'),
        (3773369, '43c7287f-7833-4ca4-966a-e11c52a3f938',
         'Inswinging — cleared', 'High pass · incomplete'),
        (3764440, '0a91219c-68ed-46c6-8a72-a2988f7a1c05',
         'Short corner — played along the ground', 'Ground pass · complete'),
    ]

    fig = plt.figure(figsize=(13.5, 11.0), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 0.34, 1, 0.34],
                          hspace=0.45, wspace=0.10,
                          left=0.055, right=0.975, top=0.82, bottom=0.05)
    for i, (mid, uuid, title, note) in enumerate(corners):
        r, c = (0, i) if i < 2 else (2, i - 2)
        panel(fig.add_subplot(gs[r, c]), fig.add_subplot(gs[r + 1, c]),
              mid, uuid, title, note)

    fig.text(0.055, 0.962, 'Reconstructed ball trajectories over real corner kicks',
             fontsize=17, color=INK)
    fig.text(0.055, 0.937,
             'Player positions and the ball\'s endpoints, flight time and swing '
             'direction are observed in StatsBomb 360 data. The path between the '
             'endpoints is reconstructed:',
             fontsize=9.5, color=INK_2)
    fig.text(0.055, 0.920,
             'ballistic height from the measured flight time, solved to arrive at '
             'the height the ball was met (headers ~2.2 yd), with a lateral bow '
             'signed by the recorded technique. Goal attacked is at the top.',
             fontsize=9.5, color=INK_2)

    handles = [
        plt.Line2D([], [], marker='o', ls='', ms=8, color=ATTACK,
                   mec=SURFACE, mew=2, label='Attacking player'),
        plt.Line2D([], [], marker='o', ls='', ms=8, color=DEFEND,
                   mec=SURFACE, mew=2, label='Defending player'),
        plt.Line2D([], [], color=BALL, lw=2.6, label='Ball path (thicker = higher)'),
        plt.Line2D([], [], marker='X', ls='', ms=10, color=BALL,
                   mec=SURFACE, mew=1.6, label='Delivery endpoint'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.055, 0.893),
               frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=4,
               columnspacing=2.2, handletextpad=0.6)

    out = os.path.join(outdir, 'corner_trajectories.png')
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print('wrote', out)
    return out


def aggregate(outdir='.', per_facet=200):
    """All reconstructed corners, faceted by delivery technique."""
    import pandas as pd
    from common import PROCESSED

    df = pd.read_parquet(f'{PROCESSED}/corner_trajectories.parquet',
                         columns=['uuid', 'point_idx', 'x', 'y', 'technique'])
    facets = [('Inswinging', ATTACK), ('Outswinging', DEFEND), ('Straight', BALL)]

    fig = plt.figure(figsize=(15.0, 5.0), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.02, right=0.98,
                          top=0.70, bottom=0.04)

    for i, (tech, colour) in enumerate(facets):
        ax = fig.add_subplot(gs[0, i])
        draw_pitch(ax)
        ax.set_anchor('N')  # sit under the title rather than floating mid-box
        sub = df[df.technique == tech]
        uuids = sub['uuid'].drop_duplicates()
        n_total = len(uuids)
        shown = uuids.head(per_facet)
        for _, g in sub[sub.uuid.isin(set(shown))].groupby('uuid', sort=False):
            ax.plot(g['y'].values, g['x'].values, '-', color=colour,
                    lw=0.8, alpha=0.22, solid_capstyle='round', zorder=3)
        ax.set_title(f'{tech}\n{n_total:,} corners  ·  {len(shown)} drawn',
                     fontsize=11, color=INK, loc='left', pad=8)

    fig.text(0.02, 0.955, 'Reconstructed corner deliveries by technique',
             fontsize=17, color=INK)
    fig.text(0.02, 0.895,
             'Every corner in the 426 matches with StatsBomb 360 coverage, '
             'reconstructed by ball_trajectory.py. An inswinger\'s arc bulges '
             'away from the goal so the ball arrives bending back toward it; '
             'an outswinger does the reverse.',
             fontsize=9.5, color=INK_2)
    fig.text(0.02, 0.850,
             'Bend direction is recorded in the data; its magnitude is a fixed '
             'prior, clamped so the ball never leaves the field of play. Goal '
             'being attacked is at the top.',
             fontsize=9.5, color=INK_2)

    out = os.path.join(outdir, 'corner_trajectories_by_technique.png')
    fig.savefig(out, dpi=170, facecolor=SURFACE,
                bbox_inches='tight', pad_inches=0.28)
    print('wrote', out)
    return out


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(target)
    aggregate(target)

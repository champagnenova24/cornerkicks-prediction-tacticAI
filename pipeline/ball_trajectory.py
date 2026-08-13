"""Ball trajectory reconstruction for pass-like events (corners in particular).

WHAT THE DATA SUPERVISES vs WHAT THIS MODULE ASSUMES
----------------------------------------------------
StatsBomb records the ball only at event boundaries. For a corner it gives:

    location        ball start (the corner arc)          [observed]
    end_location    where the delivery arrives           [observed]
    duration        flight time in seconds               [observed]
    pass.height     Ground / Low / High trajectory class [observed, coarse]
    pass.technique  Inswinging / Outswinging / Straight  [observed, sign only]

There are no intermediate ball positions, no z for passes, and no spin
magnitude. This module therefore *reconstructs* a path consistent with the
observed endpoints, flight time and shape class -- it does not recover an
observed trajectory. The vertical profile is genuine ballistics driven by the
measured duration; the lateral bow takes its SIGN from `technique` but its
MAGNITUDE is a free prior (`swing_yards`), unidentified by the data. Do not fit
claims to the curvature amount.

Coordinates follow the canonical attacking frame used across the pipeline:
120x80 yards, goal under attack at x=120. Heights are in yards.
"""
import numpy as np

# Gravity in pitch units: 9.81 m/s^2 -> yards/s^2.
G_YD = 9.81 / 0.9144

# Shoulder height is the spec's Low/High boundary (~1.45 m).
SHOULDER_YD = 1.45 / 0.9144

# Apex cap for a "Low Pass": below shoulder height by definition.
LOW_APEX_YD = 0.75 * SHOULDER_YD

# Height at which the delivery is met, by the body part that met it. The body
# part IS observed (93% of corners carry one within a few events, 55% headers);
# the height attached to each category is a prior. Solving the flight to land
# at z=0 regardless -- as if every corner dropped to the turf untouched --
# biases the apex low by ~30% on a typical headed delivery.
CONTACT_HEIGHT_YD = {
    'Head': 2.2,          # ~2.0 m, a jumping header meeting a cross
    'Keeper Arm': 2.6,    # ~2.4 m, keeper claiming above their head
    'Other': 1.2,         # chest / thigh / knee
    'Right Foot': 0.3,
    'Left Foot': 0.3,
    'Drop Kick': 0.3,
    'No Touch': 0.0,      # deliberately let run -> reaches the ground
}
# Uncontested or unknown: nothing intervened, so the ball reaches the ground.
DEFAULT_CONTACT_YD = 0.0


def contact_height(body_part):
    """Height in yards at which a delivery met `body_part` is intercepted."""
    if not body_part:
        return DEFAULT_CONTACT_YD
    return CONTACT_HEIGHT_YD.get(body_part, DEFAULT_CONTACT_YD)


def end_height_from_followups(followups, lookahead=4):
    """Infer contact height from the first following event carrying a body part.

    `followups` is the events immediately after the delivery, in index order.
    Ball Receipt / Pressure carry no body part, so a short lookahead is needed
    to reach the touch itself (a Clearance, Shot, Duel, ...).
    """
    for e in list(followups)[:lookahead]:
        detail = e.get(e['type']['name'].lower().replace(' ', '_').replace('*', ''))
        if isinstance(detail, dict) and detail.get('body_part'):
            return contact_height(detail['body_part']['name'])
    return DEFAULT_CONTACT_YD

PITCH_X, PITCH_Y = 120.0, 80.0
GOAL_CENTRE = np.array([120.0, 40.0])


def _vertical(s, duration, height_class, end_height):
    """Ballistic height profile over normalized time s = t/T.

    The ball leaves the corner arc on the ground and is met at `end_height`,
    so z(0) = 0 and z(T) = end_height. Solving z(t) = v0*t - g*t^2/2 gives
    v0 = end_height/T + g*T/2, and the flight is asymmetric: it peaks after
    mid-flight and arrives still descending (or still rising, if the contact
    is high and the flight short). end_height = 0 recovers the symmetric
    parabola with apex g*T^2/8.

    Returns (z array, apex height).
    """
    if height_class == 'Ground Pass':
        return np.zeros_like(s), 0.0

    h_end = max(float(end_height), 0.0)
    if height_class == 'Low Pass':
        # by definition the ball stays below shoulder height
        h_end = min(h_end, LOW_APEX_YD)

    T = duration
    v0 = h_end / T + G_YD * T / 2.0
    t = s * T
    z = v0 * t - G_YD * t ** 2 / 2.0
    z = np.maximum(z, 0.0)

    t_apex = v0 / G_YD
    apex = v0 ** 2 / (2.0 * G_YD) if t_apex <= T else h_end

    if height_class == 'Low Pass' and apex > LOW_APEX_YD:
        # Height class and flight time disagree (duration likely runs past the
        # bounce). The class is the stronger evidence, so scale the profile.
        scale = LOW_APEX_YD / apex
        z = z * scale
        apex = LOW_APEX_YD
    return z, apex


def _swing_sign(technique):
    """+1 bows toward the goal, -1 away from it, 0 for no swing.

    Orientation is not decided here: `reconstruct` builds a perpendicular that
    already points at the goal, so this only reports the technique's direction.

    An inswinger bows AWAY from the goal. With both endpoints fixed, an arc
    bulging to one side arrives with its velocity rotated toward the other; a
    delivery that arrives moving goalward must therefore bulge away from goal
    -- the ball is struck wide of the chord and swings back in. Bowing an
    inswinger toward the goal is not merely backwards, it is out of play: a
    corner starts ON the goal line, so any goalward bow leaves the field.
    """
    if technique == 'Inswinging':
        return -1.0
    if technique == 'Outswinging':
        return 1.0
    return 0.0


def _max_in_bounds_swing(line, shape, u, start, end):
    """Largest swing amplitude keeping every sampled point inside the pitch.

    Displacement is A * shape * u and so linear in A: solve each boundary for
    A and take the tightest. Bounds are widened to admit the observed
    endpoints, which can sit marginally outside (balls going out of play).
    """
    x_hi = max(PITCH_X, start[0], end[0])
    x_lo = min(0.0, start[0], end[0])
    y_hi = max(PITCH_Y, start[1], end[1])
    y_lo = min(0.0, start[1], end[1])

    limit = np.inf
    for axis, lo, hi in ((0, x_lo, x_hi), (1, y_lo, y_hi)):
        step = shape * u[axis]
        pos = step > 1e-12
        neg = step < -1e-12
        if pos.any():
            limit = min(limit, np.min((hi - line[pos, axis]) / step[pos]))
        if neg.any():
            limit = min(limit, np.min((lo - line[neg, axis]) / step[neg]))
    return max(float(limit), 0.0)


def reconstruct(start, end, duration, height_class='High Pass',
                technique=None, n_points=60, swing_yards=2.5,
                end_height=0.0):
    """Reconstruct a ball path between two observed endpoints.

    Args:
        start, end: (x, y) in the canonical attacking frame.
        duration: observed flight time in seconds (`event['duration']`).
        height_class: `pass.height` name -- Ground / Low / High Pass.
        technique: `pass.technique` name -- sets the swing direction only.
        n_points: samples along the path.
        swing_yards: max lateral deviation for a swung delivery. A PRIOR, not
            a measurement -- the data does not observe curvature magnitude.

    Returns:
        dict of float32 arrays: t (s), x, y, z (yards), plus scalars
        apex_z, speed (yd/s) and the flags used to build it.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    duration = float(duration) if duration and duration > 0 else np.nan

    s = np.linspace(0.0, 1.0, n_points)

    # Horizontal: constant speed along the chord (drag ignored).
    line = start[None, :] + s[:, None] * (end - start)[None, :]

    # Lateral: symmetric quadratic bow, peaking at mid-flight, clamped so the
    # ball never leaves the field of play.
    sign = _swing_sign(technique)
    d = end - start
    norm = np.hypot(*d)
    swing_used = 0.0
    if sign != 0.0 and norm > 1e-9:
        perp = np.array([-d[1], d[0]]) / norm
        if np.dot(perp, GOAL_CENTRE - (start + end) / 2.0) < 0:
            perp = -perp  # orient toward the goal being attacked
        u = sign * perp
        shape = 4.0 * s * (1.0 - s)
        swing_used = min(swing_yards,
                         _max_in_bounds_swing(line, shape, u, start, end))
        line = line + (swing_used * shape)[:, None] * u[None, :]

    # Vertical: ballistic flight from the turf to wherever the ball was met.
    T = duration if np.isfinite(duration) else 1.6
    z, apex = _vertical(s, T, height_class, end_height)

    chord = float(np.hypot(*(end - start)))
    return {
        't': (s * T).astype(np.float32),
        'x': line[:, 0].astype(np.float32),
        'y': line[:, 1].astype(np.float32),
        'z': z.astype(np.float32),
        'apex_z': float(apex),
        'end_z': float(z[-1]),
        'speed': chord / T if T > 0 else np.nan,
        'duration': T,
        'height_class': height_class,
        'technique': technique or '(none)',
        'swing_sign': float(sign),
        'swing_used': float(swing_used),
    }


def reconstruct_event(event, followups=(), n_points=60, swing_yards=2.5,
                      flip=False):
    """Reconstruct straight from a raw StatsBomb pass event dict.

    `followups` are the events immediately after this one, in index order;
    they identify the body part that met the delivery and therefore the height
    it was met at. Omit them and the ball is modelled as reaching the ground,
    which underestimates the apex of a headed delivery.

    Set flip=True when the event belongs to the team NOT in possession, to
    rotate it into the possession team's attacking frame (see stage 3).
    """
    p = event.get('pass', {})
    start = np.array(event['location'][:2], dtype=float)
    end = np.array(p['end_location'][:2], dtype=float)
    if flip:
        start = np.array([120.0, 80.0]) - start
        end = np.array([120.0, 80.0]) - end
    return reconstruct(
        start, end, event.get('duration'),
        height_class=p.get('height', {}).get('name', 'High Pass'),
        technique=p.get('technique', {}).get('name'),
        n_points=n_points, swing_yards=swing_yards,
        end_height=end_height_from_followups(followups),
    )


# ------------------------------------------------------------------ stage --

def main():
    """Emit reconstructed corner trajectories for the 360-covered matches.

    Output: data/processed/corner_trajectories.parquet, one row per sampled
    point per corner delivery (uuid joins to decision_points.parquet).
    """
    import json
    import os
    import sys

    import pandas as pd

    from common import EVENTS_DIR, PROCESSED

    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet',
                         columns=['match_id', 'uuid', 'play_pattern', 'decision_idx'])
    corners = dp[(dp.play_pattern == 'From Corner') & (dp.decision_idx == 0)]
    want = corners.groupby('match_id')['uuid'].apply(set).to_dict()

    rows = []
    for i, (mid, uuids) in enumerate(want.items()):
        try:
            with open(f'{EVENTS_DIR}/{mid}.json') as f:
                events = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        events.sort(key=lambda e: e['index'])
        for i, e in enumerate(events):
            if e['id'] not in uuids or e['type']['name'] != 'Pass':
                continue
            traj = reconstruct_event(e, followups=events[i + 1:i + 5])
            for k in range(len(traj['t'])):
                rows.append({
                    'match_id': mid, 'uuid': e['id'], 'point_idx': k,
                    't': traj['t'][k], 'x': traj['x'][k],
                    'y': traj['y'][k], 'z': traj['z'][k],
                    'apex_z': traj['apex_z'], 'end_z': traj['end_z'],
                    'speed': traj['speed'],
                    'height_class': traj['height_class'],
                    'technique': traj['technique'],
                })
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(want)} matches', file=sys.stderr)

    df = pd.DataFrame(rows)
    out = f'{PROCESSED}/corner_trajectories.parquet'
    df.to_parquet(out, index=False)
    n_corners = df['uuid'].nunique()
    per = df.groupby('uuid').first()
    print(f'wrote {out}: {n_corners:,} corners, {len(df):,} trajectory points')
    print(f"apex height (yd): mean {per['apex_z'].mean():.2f}")
    print(f"contact height (yd): mean {per['end_z'].mean():.2f}, "
          f"met above 1 yd: {(per['end_z'] > 1).mean()*100:.0f}%")
    print(df.groupby('uuid')['technique'].first().value_counts().to_string())


if __name__ == '__main__':
    main()

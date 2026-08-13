"""Stages 2-4: possession reconstruction, frame canonicalization, decision cleaning.

One pass over data/events/*.json (all 4,235 matches), emitting:

  data/processed/possessions.parquet  - one row per (match, possession) with
      terminal outcome labels (stage 2)
  data/processed/decisions.parquet    - one row per on-ball decision by the
      possession team, in the canonical attacking frame, with clipped time
      deltas and outcome-forward labels (stages 3-4)

Stage boundaries are the functions below: reconstruct_possessions (2),
canonicalize (3), extract_decisions + label_possession (4).
"""
import json
import os
import sys
from itertools import groupby
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import (EVENTS_DIR, PROCESSED, DECISION_TYPES, DT_CLIP, SHOT_ZONE,
                    ts_to_sec, flip, zone_of)

# Event types that never carry play information (admin bookkeeping).
ADMIN_TYPES = {'Starting XI', 'Half Start', 'Half End', 'Substitution',
               'Tactical Shift', 'Injury Stoppage', 'Player On', 'Player Off',
               'Bad Behaviour', 'Referee Ball-Drop', 'Camera On', 'Camera off'}

# How a possession ended, judged from its last non-admin event.
TERMINAL_BY_TYPE = {
    'Foul Won': 'foul_won',
    'Foul Committed': 'foul_committed',
    'Offside': 'offside',
    'Dispossessed': 'ball_lost',
    'Miscontrol': 'ball_lost',
    'Duel': 'ball_lost',
    'Interception': 'ball_lost',
    'Ball Recovery': 'ball_lost',
    'Block': 'ball_lost',
    'Clearance': 'ball_lost',
    'Pressure': 'ball_lost',
    '50/50': 'ball_lost',
    'Shield': 'ball_lost',
    'Error': 'ball_lost',
    'Own Goal For': 'goal',
    'Own Goal Against': 'goal_against',
}


# ---------------------------------------------------------------- stage 2 --

def reconstruct_possessions(events):
    """Group the index-ordered event stream into possession sequences."""
    events.sort(key=lambda e: e['index'])
    for pos_id, group in groupby(events, key=lambda e: e['possession']):
        yield pos_id, list(group)


# ---------------------------------------------------------------- stage 3 --

def canonicalize(event, possession_team_id):
    """Return (x, y, end_x, end_y, flipped) in the possession team's attacking
    frame (goal under attack at x=120).

    StatsBomb coordinates are normalized per *event* team, so events by the
    non-possessing team (pressures, blocks, ...) must be rotated 180 degrees
    to share the possession frame.
    """
    loc = event.get('location')
    if loc is None:
        return None, None, None, None, False
    x, y = float(loc[0]), float(loc[1])

    tname = event['type']['name']
    detail = event.get(tname.lower().replace(' ', '_'), {})
    end = detail.get('end_location') if isinstance(detail, dict) else None
    if end is not None:
        ex, ey = float(end[0]), float(end[1])  # drop z for shots here
    else:
        ex, ey = x, y  # stationary actions (dribbles) end where they start

    flipped = event['team']['id'] != possession_team_id
    if flipped:
        x, y = flip(x, y)
        ex, ey = flip(ex, ey)
    return x, y, ex, ey, flipped


# ---------------------------------------------------------------- stage 4 --

def label_possession(pos_events, possession_team_id):
    """Terminal outcome, goal flag and xG totals for one possession."""
    xg_sum, goal = 0.0, False
    for e in pos_events:
        if e['type']['name'] == 'Shot' and e['team']['id'] == possession_team_id:
            xg_sum += e['shot'].get('statsbomb_xg', 0.0)
            if e['shot'].get('outcome', {}).get('name') == 'Goal':
                goal = True

    if goal:
        return 'goal', goal, xg_sum

    terminal = 'other'
    for e in reversed(pos_events):
        tname = e['type']['name']
        if tname in ADMIN_TYPES:
            if tname == 'Half End':
                terminal = 'end_of_period'
                break
            continue
        if tname == 'Shot':
            terminal = 'shot'
        elif tname == 'Pass':
            out = e['pass'].get('outcome', {}).get('name')
            terminal = {'Out': 'out', 'Injury Clearance': 'out',
                        'Pass Offside': 'offside'}.get(out, 'ball_lost')
        elif tname in ('Ball Receipt*', 'Goal Keeper'):
            # derived / mirror events paired to the true terminal action
            continue
        else:
            terminal = TERMINAL_BY_TYPE.get(tname, 'other')
        break
    return terminal, goal, xg_sum


def extract_decisions(pos_events, possession_team_id):
    """The cleaned decision sequence: on-ball choices by the possession team.

    Drops derived events (Ball Receipt), opponent annotations (Pressure,
    Block, ... -- their signal survives as under_pressure), and admin events.
    """
    decisions = []
    for e in pos_events:
        tname = e['type']['name']
        if tname not in DECISION_TYPES or e['team']['id'] != possession_team_id:
            continue
        if e.get('location') is None:
            continue
        x, y, ex, ey, _ = canonicalize(e, possession_team_id)

        detail = e.get(tname.lower(), {})
        if tname == 'Pass':
            atype = 'cross' if detail.get('cross') else 'pass'
            success = 'outcome' not in detail
            xg = 0.0
        elif tname == 'Carry':
            atype, success, xg = 'carry', True, 0.0
        elif tname == 'Dribble':
            atype = 'dribble'
            success = detail.get('outcome', {}).get('name') == 'Complete'
            xg = 0.0
        else:  # Shot
            atype = 'shot'
            success = detail.get('outcome', {}).get('name') == 'Goal'
            xg = detail.get('statsbomb_xg', 0.0)

        decisions.append({
            'uuid': e['id'],
            'period': e['period'],
            't': ts_to_sec(e['timestamp']),
            'type': atype,
            'x': x, 'y': y, 'end_x': ex, 'end_y': ey,
            'zone': SHOT_ZONE if atype == 'shot' else zone_of(ex, ey),
            'success': success,
            'xg': xg,
            'under_pressure': bool(e.get('under_pressure', False)),
            'off_camera': bool(e.get('off_camera', False)),
            'player_id': e.get('player', {}).get('id', -1),
        })

    # time deltas between consecutive decisions, clipped past dead-ball range
    prev_t, prev_p = None, None
    for d in decisions:
        raw = d['t'] - prev_t if (prev_t is not None and d['period'] == prev_p) else 0.0
        raw = max(raw, 0.0)
        d['dt_raw'] = raw
        d['dt'] = min(raw, DT_CLIP)
        prev_t, prev_p = d['t'], d['period']

    # outcome-forward labels (value of the possession from this decision on)
    xg_after, goal_after = 0.0, False
    for i in range(len(decisions) - 1, -1, -1):
        xg_after += decisions[i]['xg']
        goal_after = goal_after or (decisions[i]['type'] == 'shot' and decisions[i]['success'])
        decisions[i]['xg_after'] = xg_after
        decisions[i]['goal_after'] = goal_after
        decisions[i]['decision_idx'] = i
        decisions[i]['n_remaining'] = len(decisions) - 1 - i
    return decisions


# ------------------------------------------------------------------ driver --

def process_match(fname):
    match_id = int(fname.replace('.json', ''))
    with open(os.path.join(EVENTS_DIR, fname)) as f:
        events = json.load(f)

    poss_rows, dec_rows = [], []
    for pos_id, pos_events in reconstruct_possessions(events):
        team = pos_events[0]['possession_team']
        terminal, goal, xg_sum = label_possession(pos_events, team['id'])
        decisions = extract_decisions(pos_events, team['id'])

        times = [ts_to_sec(e['timestamp']) for e in pos_events]
        poss_rows.append({
            'match_id': match_id, 'possession': pos_id,
            'team_id': team['id'], 'team': team['name'],
            'play_pattern': pos_events[0]['play_pattern']['name'],
            'period': pos_events[0]['period'],
            'start_t': times[0], 'duration': max(times) - times[0],
            'n_events': len(pos_events), 'n_decisions': len(decisions),
            'terminal_outcome': terminal, 'goal': goal, 'xg_sum': xg_sum,
        })
        for d in decisions:
            d['match_id'] = match_id
            d['possession'] = pos_id
            d['team_id'] = team['id']
            d['play_pattern'] = pos_events[0]['play_pattern']['name']
            d['terminal_outcome'] = terminal
            dec_rows.append(d)
    return poss_rows, dec_rows


DEC_COLS = ['match_id', 'possession', 'decision_idx', 'uuid', 'team_id',
            'player_id', 'period', 't', 'dt', 'dt_raw', 'type', 'x', 'y',
            'end_x', 'end_y', 'zone', 'success', 'xg', 'under_pressure',
            'off_camera', 'play_pattern', 'n_remaining', 'xg_after',
            'goal_after', 'terminal_outcome']


def main():
    files = sorted(f for f in os.listdir(EVENTS_DIR) if f.endswith('.json'))
    poss_writer = dec_writer = None
    poss_buf, dec_buf = [], []

    def flush():
        nonlocal poss_writer, dec_writer
        if not poss_buf:
            return
        pt = pa.Table.from_pandas(pd.DataFrame(poss_buf), preserve_index=False)
        dt = pa.Table.from_pandas(pd.DataFrame(dec_buf)[DEC_COLS], preserve_index=False)
        if poss_writer is None:
            poss_writer = pq.ParquetWriter(f'{PROCESSED}/possessions.parquet', pt.schema)
            dec_writer = pq.ParquetWriter(f'{PROCESSED}/decisions.parquet', dt.schema)
        poss_writer.write_table(pt)
        dec_writer.write_table(dt)
        poss_buf.clear()
        dec_buf.clear()

    n_poss = n_dec = 0
    with Pool(8) as pool:
        for i, (pr, dr) in enumerate(pool.imap(process_match, files, chunksize=8)):
            poss_buf.extend(pr)
            dec_buf.extend(dr)
            n_poss += len(pr)
            n_dec += len(dr)
            if len(dec_buf) >= 500_000:
                flush()
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(files)} matches', file=sys.stderr)
    flush()
    poss_writer.close()
    dec_writer.close()
    print(f'wrote {n_poss:,} possessions, {n_dec:,} decisions from {len(files)} matches')


if __name__ == '__main__':
    main()

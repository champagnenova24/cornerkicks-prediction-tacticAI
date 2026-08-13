"""Stage 5: (state, action, outcome) triples for the 360 causal core.

For the matches with StatsBomb 360 coverage, joins each cleaned decision
(from the stage-4 logic, re-derived per match for consistency) to its
freeze frame. Emits:

  data/processed/decision_points.parquet - one row per decision that has a
      freeze frame and was on camera: action factorization (type, zone),
      outcome labels, and state summary features
  data/processed/freeze_frames.parquet   - one row per visible player per
      kept decision (the full state, for the set encoder)
  data/processed/visible_area.parquet    - camera polygon per kept decision
      (the missingness mask)
"""
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import EVENTS_DIR, THREESIXTY_DIR, PROCESSED
from stage234_possessions import reconstruct_possessions, label_possession, extract_decisions


def process_match(match_id):
    # A single unreadable file (e.g. upstream data corruption) must not abort
    # the whole run: skip it, record why, and let the driver keep going.
    try:
        with open(f'{THREESIXTY_DIR}/{match_id}.json') as f:
            frames = {fr['event_uuid']: fr for fr in json.load(f)}
        with open(f'{EVENTS_DIR}/{match_id}.json') as f:
            events = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, None, None, (match_id, str(e)[:80])

    dp_rows, ff_rows, va_rows = [], [], []
    for pos_id, pos_events in reconstruct_possessions(events):
        team_id = pos_events[0]['possession_team']['id']
        play_pattern = pos_events[0]['play_pattern']['name']
        terminal, _, _ = label_possession(pos_events, team_id)
        for d in extract_decisions(pos_events, team_id):
            fr = frames.get(d['uuid'])
            if fr is None or d['off_camera']:
                continue
            ff = fr['freeze_frame']
            n_tm = sum(1 for p in ff if p['teammate'] and not p['actor'])
            n_opp = sum(1 for p in ff if not p['teammate'])
            dp_rows.append({
                'match_id': match_id, 'possession': pos_id,
                'decision_idx': d['decision_idx'], 'uuid': d['uuid'],
                'team_id': team_id, 'player_id': d['player_id'],
                'period': d['period'], 't': d['t'], 'dt': d['dt'],
                'action_type': d['type'], 'action_zone': d['zone'],
                'x': d['x'], 'y': d['y'], 'end_x': d['end_x'], 'end_y': d['end_y'],
                'success': d['success'], 'xg': d['xg'],
                'under_pressure': d['under_pressure'],
                'play_pattern': play_pattern,
                'n_remaining': d['n_remaining'], 'xg_after': d['xg_after'],
                'goal_after': d['goal_after'], 'terminal_outcome': terminal,
                'n_visible': len(ff), 'n_teammates_visible': n_tm,
                'n_opponents_visible': n_opp,
                'keeper_visible': any(p['keeper'] and not p['teammate'] for p in ff),
                'actor_in_frame': any(p['actor'] for p in ff),
            })
            for j, p in enumerate(ff):
                ff_rows.append({
                    'match_id': match_id, 'uuid': d['uuid'], 'player_idx': j,
                    'x': float(p['location'][0]), 'y': float(p['location'][1]),
                    'teammate': p['teammate'], 'actor': p['actor'],
                    'keeper': p['keeper'],
                })
            va = fr.get('visible_area') or []
            for j in range(0, len(va) - 1, 2):
                va_rows.append({
                    'match_id': match_id, 'uuid': d['uuid'], 'vertex_idx': j // 2,
                    'x': float(va[j]), 'y': float(va[j + 1]),
                })
    return dp_rows, ff_rows, va_rows, None


def main():
    matches = pd.read_parquet(f'{PROCESSED}/matches.parquet')
    ids = sorted(matches.loc[matches['has_360'], 'match_id'])
    ids = [m for m in ids if os.path.exists(f'{THREESIXTY_DIR}/{m}.json')]

    writers = {}
    bufs = {'decision_points': [], 'freeze_frames': [], 'visible_area': []}

    def flush():
        for name, buf in bufs.items():
            if not buf:
                continue
            t = pa.Table.from_pandas(pd.DataFrame(buf), preserve_index=False)
            if name not in writers:
                writers[name] = pq.ParquetWriter(f'{PROCESSED}/{name}.parquet', t.schema)
            writers[name].write_table(t)
            buf.clear()

    counts = {k: 0 for k in bufs}
    skipped = []
    with Pool(8) as pool:
        for i, (dp, ff, va, err) in enumerate(pool.imap(process_match, ids, chunksize=4)):
            if err is not None:
                skipped.append(err)
                continue
            bufs['decision_points'].extend(dp)
            bufs['freeze_frames'].extend(ff)
            bufs['visible_area'].extend(va)
            for k, rows in zip(counts, (dp, ff, va)):
                counts[k] += len(rows)
            if len(bufs['freeze_frames']) >= 1_000_000:
                flush()
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(ids)} matches', file=sys.stderr)
    flush()
    for w in writers.values():
        w.close()
    print(f'{len(ids) - len(skipped)}/{len(ids)} matches with 360 processed:')
    for k, v in counts.items():
        print(f'  {k}: {v:,} rows')
    if skipped:
        with open(f'{PROCESSED}/stage5_skipped.txt', 'w') as f:
            for mid, msg in skipped:
                f.write(f'{mid}\t{msg}\n')
        print(f'skipped {len(skipped)} unreadable match(es); logged to '
              f'{PROCESSED}/stage5_skipped.txt')
        for mid, msg in skipped:
            print(f'  {mid}: {msg}')


if __name__ == '__main__':
    main()

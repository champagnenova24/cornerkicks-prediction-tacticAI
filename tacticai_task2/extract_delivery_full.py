"""Delivery attributes for every corner in the full StatsBomb open data.

`add_delivery.py` did this for the 2,473 corners in the 360-covered matches.
This does it for all 4,235 matches -- technique, height, body part, flight time
and the endpoints -- so the delivery analysis is not limited to the subset that
happens to carry freeze frames.

The corner set comes from `corners_full_raw.parquet` (identified in
decisions.parquet as the first decision of a From Corner possession, taken from
the corner arc; validated against the exactly-identified 360 subset, 2,472 of
2,473 recovered). Only the per-delivery attributes need the raw JSON.

12 GB of JSON, so the files are parsed in a process pool.

Output: corner_delivery_full.parquet
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
from paths import EVENTS
N_WORKERS = 8


def scan(args):
    """Pull delivery attributes for the wanted corner uuids in one match."""
    match_id, uuids = args
    try:
        with open(f'{EVENTS}/{match_id}.json') as f:
            events = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    want = set(uuids)
    out = []
    for e in events:
        if e['id'] not in want:
            continue
        p = e.get('pass', {}) or {}
        loc = e.get('location') or [None, None]
        end = p.get('end_location') or [None, None]
        out.append(dict(
            corner_id=e['id'], match_id=int(match_id),
            start_x=loc[0], start_y=loc[1],
            end_x=end[0], end_y=end[1],
            duration=e.get('duration'),
            technique=(p.get('technique') or {}).get('name', 'Missing'),
            height=(p.get('height') or {}).get('name', 'Missing'),
            body_part=(p.get('body_part') or {}).get('name', 'Missing'),
            pass_length=p.get('length'),
            pass_type=(p.get('type') or {}).get('name'),
            outcome=(p.get('outcome') or {}).get('name', 'Complete'),
        ))
    return out


def main():
    c = pd.read_parquet(f'{SP}/corners_full_raw.parquet')
    jobs = [(int(m), g.uuid.tolist())
            for m, g in c.groupby('match_id')]
    print(f'{len(c):,} corners across {len(jobs):,} matches, '
          f'{N_WORKERS} workers', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(scan, jobs, chunksize=8)):
            rows.extend(res)
            if (i + 1) % 500 == 0:
                print(f'  {i + 1}/{len(jobs)} matches, {len(rows):,} corners',
                      flush=True)

    df = pd.DataFrame(rows)
    # verified corner kicks only: pass.type is the authoritative marker
    n_all = len(df)
    df = df[df.pass_type == 'Corner'].copy()
    print(f'\npass.type == Corner: {len(df):,} of {n_all:,} extracted')
    df.to_parquet(f'{SP}/corner_delivery_full.parquet', index=False)
    for col in ('technique', 'height', 'body_part'):
        print(f'\n{col}:\n{df[col].value_counts().to_string()}')
    print(f'\nwrote corner_delivery_full.parquet')


if __name__ == '__main__':
    main()

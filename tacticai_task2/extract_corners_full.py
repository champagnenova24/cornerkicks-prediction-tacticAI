"""Identify every corner in the full StatsBomb open data, with its outcome.

The 360-subset pipeline (`extract_corners.py`) reads raw event JSON and filters
`pass.type == Corner`. That is exact but needs 12 GB of parsing. For the full
set the corners can be pulled straight out of the pipeline's
`decisions.parquet`, which already covers all 4,235 matches and carries the
outcome-forward labels:

    play_pattern == 'From Corner'   the possession began with a corner
    decision_idx == 0               this is its first decision
    x >= 119 and (y <= 1 or y >= 79) taken from the corner arc

The arc test is what makes it safe: a From Corner possession whose first
recorded decision is not on the arc is not the delivery. Validated against the
2,473 exactly-identified 360-subset corners -- see the check printed below.

Delivery attributes (technique / height / body part / flight time) still need
the raw JSON; `extract_delivery_full.py` does that, keyed on the output here.

Output: corners_full_raw.parquet
"""
import os

import pandas as pd
import pyarrow.parquet as pq

SP = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(os.path.dirname(SP), 'data', 'processed')

COLS = ['match_id', 'uuid', 'type', 'x', 'y', 'end_x', 'end_y',
        'play_pattern', 'decision_idx', 'xg_after', 'goal_after', 'off_camera']


def main():
    f = pq.ParquetFile(f'{PROCESSED}/decisions.parquet')
    keep = []
    for b in f.iter_batches(batch_size=1_000_000, columns=COLS):
        d = b.to_pandas()
        d = d[(d.play_pattern == 'From Corner') & (d.decision_idx == 0)]
        if len(d):
            keep.append(d)
    c = pd.concat(keep, ignore_index=True)
    print(f'From Corner & decision_idx == 0: {len(c):,} rows')
    print(f'  type breakdown: {dict(c.type.value_counts())}')

    on_arc = (c.x >= 119) & ((c.y <= 1) | (c.y >= 79))
    print(f'  on the corner arc: {on_arc.sum():,} ({on_arc.mean()*100:.1f}%)')
    c = c[on_arc].copy()

    # validation against the exactly-identified subset
    try:
        known = set(pd.read_parquet(f'{SP}/corners.parquet').corner_id)
        rec = len(known & set(c.uuid))
        print(f'\nvalidation: recovers {rec:,} of {len(known):,} corners '
              f'identified exactly from raw events '
              f'({rec / len(known) * 100:.1f}%)')
        if rec / len(known) < 0.98:
            print('  WARNING: recovery below 98% -- the heuristic is off')
    except FileNotFoundError:
        print('\n(corners.parquet absent, skipping validation)')

    c.to_parquet(f'{SP}/corners_full_raw.parquet', index=False)
    print(f'\nwrote corners_full_raw.parquet: {len(c):,} corners across '
          f'{c.match_id.nunique():,} matches')


if __name__ == '__main__':
    main()

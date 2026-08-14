"""Shared data-path resolution for the tacticai_task2 scripts.

Same lookup as pipeline/common.py: $OPEN_DATA first, then known locations.
"""
import os

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.environ.get('OPEN_DATA', ''),
               os.path.join(_PROJECT, 'open-data-master'),
               os.path.expanduser('~/Desktop/open-data-master'),
               os.path.expanduser('~/Downloads/open-data-master')]
try:
    OPEN_DATA = next(p for p in _CANDIDATES if p and os.path.isdir(p))
except StopIteration:
    raise SystemExit(
        'StatsBomb open-data checkout not found. Set $OPEN_DATA or clone\n'
        'https://github.com/statsbomb/open-data into the repository root as\n'
        f'{os.path.join(_PROJECT, "open-data-master")}')

DATA = os.path.join(OPEN_DATA, 'data')
EVENTS = os.path.join(DATA, 'events')
PROCESSED = os.path.join(_PROJECT, 'data', 'processed')

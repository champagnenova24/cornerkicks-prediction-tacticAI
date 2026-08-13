"""Shared constants and helpers for the StatsBomb counterfactual pipeline."""
import hashlib
import os

# StatsBomb open-data checkout: env override first, then known locations.
_CANDIDATES = [os.environ.get('OPEN_DATA', ''),
               os.path.expanduser('~/Desktop/open-data-master'),
               os.path.expanduser('~/Downloads/open-data-master')]
OPEN_DATA = next(p for p in _CANDIDATES if p and os.path.isdir(p))
EVENTS_DIR = os.path.join(OPEN_DATA, 'data', 'events')
THREESIXTY_DIR = os.path.join(OPEN_DATA, 'data', 'three-sixty')
MATCH_INDEX = os.path.join(OPEN_DATA, 'match_index.csv')

# Outputs live inside the project no matter where the project directory sits.
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(_PROJECT, 'data', 'processed')

PITCH_X, PITCH_Y = 120.0, 80.0

# Action space kept as "decisions" (on-ball choices by the possession team).
DECISION_TYPES = {'Pass', 'Carry', 'Shot', 'Dribble'}

# 12x8 grid of 10-yard cells for action targets.
ZONE_COLS, ZONE_ROWS = 12, 8
SHOT_ZONE = -1  # shots target the goal, not a pitch zone

DT_CLIP = 10.0  # seconds; ~p97 of inter-event gaps, beyond is dead-ball time

# Held out entirely for out-of-distribution evaluation.
OOD_COMPETITION = ("Women's World Cup", '2023')


def ts_to_sec(ts: str) -> float:
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def match_split(match_id) -> str:
    """Deterministic 80/10/10 train/val/test split by match id."""
    h = int(hashlib.md5(str(match_id).encode()).hexdigest(), 16) % 100
    if h < 80:
        return 'train'
    if h < 90:
        return 'val'
    return 'test'


def flip(x: float, y: float):
    """Rotate into the opposite attacking frame (180-degree pitch flip)."""
    return PITCH_X - x, PITCH_Y - y


def zone_of(x: float, y: float) -> int:
    zx = min(max(int(x // 10), 0), ZONE_COLS - 1)
    zy = min(max(int(y // 10), 0), ZONE_ROWS - 1)
    return zx * ZONE_ROWS + zy

"""The 360 camera polygon as a missingness mask.

A 360 freeze frame lists only the players the broadcast camera could see, and
`visible_area` is the polygon it could see. Those two together separate two
statements the freeze frame alone conflates:

    "there is no defender at (x, y)"       -- (x, y) is inside the polygon
                                             and no one is standing there
    "we do not know who is at (x, y)"     -- (x, y) is outside the polygon

Everything here works in the frame of the event the freeze frame belongs to
(the actor's attacking frame, which for decision points is the possession
team's canonical frame -- no flip is needed).

The polygons are clipped to the pitch (no vertex lies outside [0, 120] x
[0, 80]), so a region's visible share is measured over its ON-PITCH part only.
Otherwise a disc around a goalpost -- half of it behind the goal line -- could
never be more than 50% "seen", and every zero there would read as unobserved.

Library, imported by `tensorize.py`, `positivity.py` and
`tacticai_task2/defense_features.py`. Not run.
"""
import numpy as np
import pandas as pd
from matplotlib.path import Path

from common import PITCH_X, PITCH_Y

# Grid for the rasterized mask: 5-yard cells, 24 x 16 over the pitch.
CELL = 5.0
GRID_X, GRID_Y = int(PITCH_X / CELL), int(PITCH_Y / CELL)
_CX = (np.arange(GRID_X) + 0.5) * CELL
_CY = (np.arange(GRID_Y) + 0.5) * CELL
CELL_CENTRES = np.array([(x, y) for y in _CY for x in _CX])   # row-major (y, x)


def polygons(va):
    """visible_area rows (uuid, vertex_idx, x, y) -> {uuid: (n, 2) array}."""
    va = va.sort_values(['uuid', 'vertex_idx'])
    xy = va[['x', 'y']].to_numpy(float)
    out = {}
    uu = va.uuid.to_numpy()
    if len(uu) == 0:
        return out
    starts = np.r_[0, np.flatnonzero(uu[1:] != uu[:-1]) + 1, len(uu)]
    for a, b in zip(starts[:-1], starts[1:]):
        out[uu[a]] = xy[a:b]
    return out


def contains(poly, pts):
    """Boolean mask: which of `pts` (m, 2) lie inside polygon `poly` (n, 2).

    A polygon with fewer than 3 vertices has no area: nothing is visible.
    """
    if poly is None or len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    return Path(poly).contains_points(pts)


def grid_mask(poly):
    """(GRID_Y, GRID_X) bool: cell centre is inside the camera polygon."""
    return contains(poly, CELL_CENTRES).reshape(GRID_Y, GRID_X)


def _on_pitch(pts):
    return pts[(pts[:, 0] > 0) & (pts[:, 0] < PITCH_X)
               & (pts[:, 1] > 0) & (pts[:, 1] < PITCH_Y)]


def region_visible_frac(poly, region, step=1.0):
    """Share of the on-pitch part of a rectangle (x0, x1, y0, y1) inside the
    polygon, by a `step`-yard lattice. 1.0 = fully seen, so a zero count there
    is a real zero; below 1 the count is a lower bound."""
    x0, x1, y0, y1 = region
    gx = np.arange(x0 + step / 2, x1, step)
    gy = np.arange(y0 + step / 2, y1, step)
    pts = _on_pitch(np.array([(x, y) for y in gy for x in gx]).reshape(-1, 2))
    return float(contains(poly, pts).mean()) if len(pts) else np.nan


def disc_visible_frac(poly, cx, cy, r, n=64):
    """Share of the on-pitch part of a disc of radius r (yd) around (cx, cy)
    inside the polygon, sampled on a sunflower lattice of n points."""
    k = np.arange(n) + 0.5
    rad = r * np.sqrt(k / n)
    ang = np.pi * (1 + 5 ** 0.5) * k
    pts = _on_pitch(np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
    return float(contains(poly, pts).mean()) if len(pts) else np.nan


def load_polygons(path, uuids=None):
    """Read visible_area.parquet (optionally only some uuids) into polygons."""
    import pyarrow.parquet as pq
    want = None if uuids is None else set(uuids)
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1_000_000, columns=['uuid', 'vertex_idx', 'x', 'y']):
        d = batch.to_pandas()
        if want is not None:
            d = d[d.uuid.isin(want)]
        if len(d):
            parts.append(d)
    if not parts:
        return {}
    return polygons(pd.concat(parts, ignore_index=True))

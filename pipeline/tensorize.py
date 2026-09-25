"""Stage 6: tensorize the 360 causal core for a set-encoder model.

Stages 1-5 leave one row per decision point, one row per visible player and
one row per camera-polygon vertex. A model wants fixed-shape arrays. This
stage produces them, one compressed .npz per split:

  state      players     (N, P_MAX, 5)  x/120, y/80, teammate, actor, keeper
             player_mask (N, P_MAX)     slot holds a visible player
             visible     (N, GY, GX)    5-yd cell centre inside the camera
                                        polygon -- the missingness mask
             ctx         (N, 4)         ball x/120, y/80, under_pressure, dt/10
  action     action_type (N,)           index into ACTION_TYPES
             action_zone (N,)           12x8 zone of the target, -1 = shot
             action_end  (N, 2)         end x/120, y/80
  continuation
             cont        (N, H, 6)      next H decisions of the SAME
                                        possession: type, x, y, end_x, end_y
                                        (normalized), dt/10
             cont_mask   (N, H)         real step vs padding
             truncated   (N,)           the possession continues past H
  outcome    xg_after, goal_after, terminal (index into TERMINALS)
  keys       uuid, match_id, possession, decision_idx

HOW `visible` IS MEANT TO BE USED. A set encoder over `players` with
`player_mask` as key-padding mask sees only who is there. That silently reads
every unseen region as empty, and the 360 camera covers roughly three
quarters of the players. Adding one token per grid cell -- carrying the cell
centre and the count of visible players in it -- and masking out the cells
where `visible` is False gives the model "empty and seen" and "unknown" as
different inputs, which is the whole point of shipping the polygon. The
attention mask over cell tokens is exactly `visible.reshape(N, -1)`.

The continuation is taken from `decisions.parquet`, so it includes steps that
were off camera (they have no freeze frame of their own but they did happen).
Padding is zeros with `cont_mask = False`. Possessions are long-tailed (median
7 decisions remain after a decision point, p90 28), so no horizon covers them
all: H = 16 holds the whole remainder for 76% of decision points (H = 8 would
manage 55%), and `truncated` marks the rest so a sequence model can tell "the
possession ended" from "the window ended". The outcome labels (`xg_after`,
`goal_after`, `terminal`) always cover the whole possession.

Run after stage 5:   python tensorize.py
Output:              data/processed/tensors/{train,val,test,ood_test}.npz
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import PROCESSED, PITCH_X, PITCH_Y, DT_CLIP
from visibility import GRID_X, GRID_Y, grid_mask, load_polygons

P_MAX = 22
H = 16
ACTION_TYPES = ['pass', 'carry', 'cross', 'dribble', 'shot']
TERMINALS = ['goal', 'shot', 'ball_lost', 'out', 'foul_won', 'foul_committed',
             'offside', 'end_of_period', 'goal_against', 'other']
OUT = os.path.join(PROCESSED, 'tensors')


def load_points():
    dp = pd.read_parquet(f'{PROCESSED}/decision_points.parquet', columns=[
        'match_id', 'possession', 'decision_idx', 'uuid', 'action_type',
        'action_zone', 'x', 'y', 'end_x', 'end_y', 'dt', 'under_pressure',
        'xg_after', 'goal_after', 'terminal_outcome'])
    dp['action_type'] = dp.action_type.astype(str)
    dp['terminal_outcome'] = dp.terminal_outcome.astype(str)
    m = pd.read_parquet(f'{PROCESSED}/matches.parquet',
                        columns=['match_id', 'split'])
    dp = dp.merge(m, on='match_id', how='left')
    return dp.sort_values(['match_id', 'possession', 'decision_idx']) \
             .reset_index(drop=True)


def continuations(dp):
    """Next H decisions of the same possession, gathered by sorted position."""
    mids = dp.match_id.unique().tolist()
    dec = pq.read_table(f'{PROCESSED}/decisions.parquet',
                        columns=['match_id', 'possession', 'decision_idx',
                                 'type', 'x', 'y', 'end_x', 'end_y', 'dt'],
                        filters=[('match_id', 'in', mids)]).to_pandas()
    dec['type'] = dec['type'].astype(str)
    dec = dec.sort_values(['match_id', 'possession', 'decision_idx']) \
             .reset_index(drop=True)
    pos = dec.reset_index()[['index', 'match_id', 'possession', 'decision_idx']]
    j = dp[['match_id', 'possession', 'decision_idx']].merge(
        pos, on=['match_id', 'possession', 'decision_idx'], how='left')
    if j['index'].isna().any():
        raise SystemExit(f'{int(j["index"].isna().sum())} decision points have '
                         'no row in decisions.parquet -- rerun stages 2-5')
    base = j['index'].to_numpy(int)

    tcode = dec['type'].map({a: i for i, a in enumerate(ACTION_TYPES)}) \
                       .fillna(-1).to_numpy(np.float32)
    feats = np.c_[tcode, dec.x / PITCH_X, dec.y / PITCH_Y,
                  dec.end_x / PITCH_X, dec.end_y / PITCH_Y,
                  dec.dt / DT_CLIP].astype(np.float32)
    key_m = dec.match_id.to_numpy()
    key_p = dec.possession.to_numpy()
    n = len(dp)
    cont = np.zeros((n, H, feats.shape[1]), np.float32)
    mask = np.zeros((n, H), bool)
    for h in range(1, H + 1):
        p = base + h
        ok = p < len(dec)
        pc = np.where(ok, p, 0)
        ok &= (key_m[pc] == key_m[base]) & (key_p[pc] == key_p[base])
        cont[ok, h - 1] = feats[pc[ok]]
        mask[ok, h - 1] = True
    # whole remaining possession fits in the horizon?
    after = dec.groupby(['match_id', 'possession']).decision_idx.transform('max') \
               .to_numpy()[base] - dp.decision_idx.to_numpy()
    return cont, mask, after


def freeze_frames(dp):
    row = pd.Series(np.arange(len(dp)), index=dp.uuid.to_numpy())
    players = np.zeros((len(dp), P_MAX, 5), np.float32)
    pmask = np.zeros((len(dp), P_MAX), bool)
    clipped = 0
    for b in pq.ParquetFile(f'{PROCESSED}/freeze_frames.parquet').iter_batches(
            batch_size=2_000_000, columns=['uuid', 'player_idx', 'x', 'y',
                                           'teammate', 'actor', 'keeper']):
        d = b.to_pandas()
        r = row.reindex(d.uuid.to_numpy()).to_numpy()
        ok = ~np.isnan(r)
        k = d.player_idx.to_numpy()
        clipped += int((ok & (k >= P_MAX)).sum())
        ok &= k < P_MAX
        r, k = r[ok].astype(int), k[ok]
        d = d[ok]
        players[r, k] = np.c_[d.x / PITCH_X, d.y / PITCH_Y, d.teammate,
                              d.actor, d.keeper].astype(np.float32)
        pmask[r, k] = True
    return players, pmask, clipped


def visibility(dp):
    polys = load_polygons(f'{PROCESSED}/visible_area.parquet')
    vis = np.zeros((len(dp), GRID_Y, GRID_X), bool)
    t0 = time.time()
    missing = 0
    for i, u in enumerate(dp.uuid.to_numpy()):
        p = polys.get(u)
        if p is None:
            missing += 1
            continue
        vis[i] = grid_mask(p)
        if (i + 1) % 200_000 == 0:
            print(f'  rasterized {i + 1:,} polygons ({time.time() - t0:.0f}s)',
                  file=sys.stderr, flush=True)
    return vis, missing


def main():
    os.makedirs(OUT, exist_ok=True)
    dp = load_points()
    print(f'{len(dp):,} decision points in {dp.match_id.nunique()} matches')

    cont, cmask, after = continuations(dp)
    print(f'continuation horizon H={H}: whole remaining possession fits for '
          f'{(after <= H).mean()*100:.1f}% of decision points '
          f'(median {np.median(after):.0f} decisions remain)')

    players, pmask, clipped = freeze_frames(dp)
    print(f'freeze frames: median {np.median(pmask.sum(1)):.0f} visible players, '
          f'{clipped} player rows beyond P_MAX={P_MAX} dropped')

    vis, missing = visibility(dp)
    print(f'camera polygon: median {np.median(vis.mean((1, 2)))*100:.0f}% of the '
          f'pitch visible, {missing} decision points without a polygon')

    atype = dp.action_type.map({a: i for i, a in enumerate(ACTION_TYPES)})
    term = dp.terminal_outcome.map({a: i for i, a in enumerate(TERMINALS)})
    if atype.isna().any() or term.isna().any():
        raise SystemExit(f'unmapped labels: {set(dp.action_type[atype.isna()])} '
                         f'{set(dp.terminal_outcome[term.isna()])}')
    arrays = dict(
        players=players, player_mask=pmask, visible=vis,
        ctx=np.c_[dp.x / PITCH_X, dp.y / PITCH_Y, dp.under_pressure,
                  dp.dt / DT_CLIP].astype(np.float32),
        action_type=atype.to_numpy(np.int8),
        action_zone=dp.action_zone.to_numpy(np.int16),
        action_end=np.c_[dp.end_x / PITCH_X, dp.end_y / PITCH_Y].astype(np.float32),
        cont=cont, cont_mask=cmask, truncated=after > H,
        xg_after=dp.xg_after.to_numpy(np.float32),
        goal_after=dp.goal_after.to_numpy(bool),
        terminal=term.to_numpy(np.int8),
        uuid=dp.uuid.to_numpy().astype('U36'),
        match_id=dp.match_id.to_numpy(np.int64),
        possession=dp.possession.to_numpy(np.int32),
        decision_idx=dp.decision_idx.to_numpy(np.int32),
    )
    print(f'\n{"split":<10}{"N":>10}{"matches":>9}   file')
    for split, idx in dp.groupby('split').indices.items():
        path = os.path.join(OUT, f'{split}.npz')
        np.savez_compressed(path, **{k: v[idx] for k, v in arrays.items()},
                            action_types=np.array(ACTION_TYPES),
                            terminals=np.array(TERMINALS))
        print(f'{split:<10}{len(idx):>10,}{dp.match_id.iloc[idx].nunique():>9}   '
              f'{os.path.relpath(path, PROCESSED)} '
              f'({os.path.getsize(path)/1e6:.0f} MB)')


if __name__ == '__main__':
    main()

"""Build a TacticAI-Task-2-style corner kick dataset from StatsBomb open data.

For every corner kick in a match with 360 freeze-frame coverage:
  - snapshot of visible players at the moment the corner is taken (normalized
    so the attacking goal is at (120, 40) and the corner is taken from (120, 0))
  - receiver label: first player to touch the ball after the corner,
    matched to the nearest freeze-frame player with the correct team flag
  - shot label (paper definition): the first ball touch by the attacking team
    after the corner is a shot attempt (incl. first-time shots recorded as
    receipt + shot, via the corner pass's shot_assist/goal_assist flags)
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MAX_SPEED = 9.5  # m/s cap for plausible player movement

DATA = os.path.expanduser("/Users/daniel/Desktop/open-data-master/data")
OUT = os.path.dirname(os.path.abspath(__file__))

# event types that imply the player touched the ball
TOUCH_TYPES = {
    "Ball Receipt*", "Pass", "Shot", "Clearance", "Interception", "Dribble",
    "Miscontrol", "Ball Recovery", "Block", "Own Goal Against",
}
GK_TOUCH = {"Claim", "Collected", "Punch", "Save", "Shot Saved", "Smother",
            "Keeper Sweeper", "Shot Saved Off T", "Shot Saved To Post",
            "Saved To Post"}
RESTART_PASS = {"Throw-in", "Goal Kick", "Free Kick", "Kick Off", "Corner"}


def ts_seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def is_touch(e):
    t = e["type"]["name"]
    if t not in TOUCH_TYPES and t != "Goal Keeper":
        return False
    if t == "Ball Receipt*":
        # incomplete receipt = ball never arrived
        return e.get("ball_receipt", {}).get("outcome", {}).get("name") != "Incomplete"
    if t == "Goal Keeper":
        return e.get("goalkeeper", {}).get("type", {}).get("name") in GK_TOUCH
    return True


def is_restart(e):
    if e["type"]["name"] in ("Half End", "Half Start"):
        return True
    if e["type"]["name"] == "Pass" and \
            e.get("pass", {}).get("type", {}).get("name") in RESTART_PASS:
        return True
    return False


def flip(loc):
    return [120.0 - loc[0], 80.0 - loc[1]]


def estimate_velocities(frame0, frame1, flip_f1, dt):
    """Match anonymous players between two freeze frames (per team side,
    keepers first) and return per-frame0-player (vx, vy, has_vel) in the
    frame0 orientation. frame1 coords/teammate flags are flipped when the
    second event belongs to the other team."""
    p1 = []
    for p in frame1["freeze_frame"]:
        loc = flip(p["location"]) if flip_f1 else p["location"][:2]
        p1.append(dict(loc=np.asarray(loc, dtype=float),
                       teammate=p["teammate"] ^ flip_f1, keeper=p["keeper"]))
    out = [(0.0, 0.0, False)] * len(frame0["freeze_frame"])
    cap = MAX_SPEED * dt + 2.0
    for side in (True, False):
        for gk in (True, False):
            i0 = [i for i, p in enumerate(frame0["freeze_frame"])
                  if p["teammate"] == side and p["keeper"] == gk]
            i1 = [j for j, p in enumerate(p1)
                  if p["teammate"] == side and p["keeper"] == gk]
            if not i0 or not i1:
                continue
            P0 = np.array([frame0["freeze_frame"][i]["location"][:2] for i in i0])
            P1 = np.array([p1[j]["loc"] for j in i1])
            D = np.linalg.norm(P0[:, None, :] - P1[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(D)
            for r, c in zip(rows, cols):
                if D[r, c] > cap:
                    continue
                v = (P1[c] - P0[r]) / dt
                s = np.linalg.norm(v)
                if s > MAX_SPEED:
                    v *= MAX_SPEED / s
                out[i0[r]] = (float(v[0]), float(v[1]), True)
    return out


def process_match(match_id):
    with open(f"{DATA}/events/{match_id}.json") as f:
        events = json.load(f)
    with open(f"{DATA}/three-sixty/{match_id}.json") as f:
        frames = {fr["event_uuid"]: fr for fr in json.load(f)}
    events.sort(key=lambda e: e["index"])

    corners, players = [], []
    for i, e in enumerate(events):
        if e["type"]["name"] != "Pass" or \
                e.get("pass", {}).get("type", {}).get("name") != "Corner":
            continue
        frame = frames.get(e["id"])
        if frame is None or "location" not in e:
            continue
        att_team = e["team"]["id"]
        t0, period = ts_seconds(e["timestamp"]), e["period"]
        corner_y = e["location"][1]
        # normalize so every corner is taken from the (120, 0) corner spot
        reflect = corner_y > 40

        # receiver = first touch by anyone; shot label from first attacking touch
        receiver_ev, first_att_ev = None, None
        for n in events[i + 1:i + 60]:
            if n["period"] != period:
                break
            if ts_seconds(n["timestamp"]) - t0 > 25:
                break
            if is_restart(n):
                break
            if not is_touch(n) or "player" not in n:
                continue
            if receiver_ev is None:
                receiver_ev = n
            if n["team"]["id"] == att_team:
                first_att_ev = n
                break

        shot = 0
        if first_att_ev is not None:
            tname = first_att_ev["type"]["name"]
            if tname == "Shot":
                shot = 1
            elif tname == "Ball Receipt*" and (
                    e["pass"].get("shot_assist") or e["pass"].get("goal_assist")):
                shot = 1

        cid = e["id"]

        # ---- velocity from the next event that carries a freeze frame ----
        vels, dt_flight = None, None
        for n in events[i + 1:i + 15]:
            if n["period"] != period:
                break
            nf = frames.get(n["id"])
            if nf is None:
                continue
            dt = ts_seconds(n["timestamp"]) - t0
            if 0.3 <= dt <= 6.0:
                vels = estimate_velocities(
                    frame, nf, n["team"]["id"] != att_team, dt)
                dt_flight = dt
            break

        n_att = n_def = 0
        cand_rows = []
        for j, p in enumerate(frame["freeze_frame"]):
            x, y = p["location"][:2]
            vx, vy, has_vel = vels[j] if vels else (0.0, 0.0, False)
            if reflect:
                y = 80.0 - y
                vy = -vy
            cand_rows.append(dict(
                corner_id=cid, x=x, y=y, teammate=p["teammate"],
                actor=p["actor"], keeper=p["keeper"], is_receiver=False,
                vx=vx, vy=vy, has_vel=has_vel))
            if p["teammate"]:
                n_att += 1
            else:
                n_def += 1

        # ---- receiver matching ----
        recv_idx, recv_dist = None, None
        if receiver_ev is not None:
            same_team = receiver_ev["team"]["id"] == att_team
            loc = receiver_ev.get("location")
            if loc is not None:
                rx, ry = loc[:2]
                if not same_team:
                    rx, ry = flip([rx, ry])
                if reflect:
                    ry = 80.0 - ry
                best, bestd = None, 1e9
                for j, (p, row) in enumerate(zip(frame["freeze_frame"], cand_rows)):
                    if p["teammate"] != same_team or p["actor"]:
                        continue
                    d = ((row["x"] - rx) ** 2 + (row["y"] - ry) ** 2) ** 0.5
                    if d < bestd:
                        best, bestd = j, d
                if best is not None:
                    recv_idx, recv_dist = best, bestd
                    cand_rows[best]["is_receiver"] = True

        corners.append(dict(
            match_id=match_id, corner_id=cid, index=e["index"], period=period,
            timestamp=e["timestamp"], att_team=att_team,
            competition=None,  # filled later from matches parquet if needed
            corner_y_raw=corner_y, reflected=reflect,
            n_att_visible=n_att, n_def_visible=n_def,
            shot=shot,
            receiver_found=receiver_ev is not None,
            receiver_is_teammate=(receiver_ev["team"]["id"] == att_team)
            if receiver_ev is not None else None,
            receiver_event_type=receiver_ev["type"]["name"]
            if receiver_ev is not None else None,
            receiver_match_dist=recv_dist,
            receiver_player_id=receiver_ev["player"]["id"]
            if receiver_ev is not None else None,
            receiver_player_name=receiver_ev["player"]["name"]
            if receiver_ev is not None else None,
            dt_flight=dt_flight,
            vel_coverage=float(np.mean([r["has_vel"] for r in cand_rows]))
            if cand_rows else 0.0,
        ))
        players.extend(cand_rows)
    return corners, players


def main():
    match_ids = [f[:-5] for f in os.listdir(f"{DATA}/three-sixty")
                 if f.endswith(".json")]
    all_c, all_p = [], []
    for k, mid in enumerate(sorted(match_ids)):
        try:
            c, p = process_match(mid)
            all_c.extend(c)
            all_p.extend(p)
        except Exception as ex:
            print(f"skip {mid}: {ex}")
    dc = pd.DataFrame(all_c)
    dp = pd.DataFrame(all_p)
    dc.to_parquet(f"{OUT}/corners.parquet")
    dp.to_parquet(f"{OUT}/corner_players.parquet")
    print(f"matches: {len(match_ids)}  corners: {len(dc)}  "
          f"shot rate: {dc.shot.mean():.3f}")
    print(f"receiver found: {dc.receiver_found.mean():.3f}  "
          f"match dist median: {dc.receiver_match_dist.median():.2f}m  "
          f"p90: {dc.receiver_match_dist.quantile(.9):.2f}m")
    print(dc.receiver_event_type.value_counts().head(10))


if __name__ == "__main__":
    main()

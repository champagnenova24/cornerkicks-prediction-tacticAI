"""Reconstruct who is on the pitch (and who is in goal) at every corner.

Output: onpitch.parquet — one row per corner with attacking/defending player
id lists and GK ids, joinable to corners.parquet on corner_id.
"""
import json
import os
import pandas as pd

from paths import DATA
OUT = os.path.dirname(os.path.abspath(__file__))
RED = {"Red Card", "Second Yellow"}


def process_match(match_id, corner_rows):
    events = json.load(open(f"{DATA}/events/{match_id}.json"))
    events.sort(key=lambda e: e["index"])
    corners = {r.index: r.corner_id for r in corner_rows.itertuples()}
    want = dict(sorted(corners.items()))

    onpitch, gk = {}, {}
    out = []
    for e in events:
        t = e["type"]["name"]
        team = e.get("team", {}).get("id")
        if t == "Starting XI":
            lineup = e["tactics"]["lineup"]
            onpitch[team] = {p["player"]["id"] for p in lineup}
            for p in lineup:
                if p["position"]["name"] == "Goalkeeper":
                    gk[team] = p["player"]["id"]
        elif t == "Tactical Shift":
            for p in e["tactics"]["lineup"]:
                if p["position"]["name"] == "Goalkeeper":
                    gk[team] = p["player"]["id"]
        elif t == "Substitution":
            off = e["player"]["id"]
            on = e["substitution"]["replacement"]["id"]
            onpitch[team].discard(off)
            onpitch[team].add(on)
            if gk.get(team) == off:
                gk[team] = on
        elif t in ("Foul Committed", "Bad Behaviour"):
            card = (e.get("foul_committed", {}).get("card", {}) or
                    e.get("bad_behaviour", {}).get("card", {})).get("name")
            if card in RED and "player" in e:
                onpitch[team].discard(e["player"]["id"])

        if e["index"] in want:
            att = e["team"]["id"]
            deft = next((tm for tm in onpitch if tm != att), None)
            out.append(dict(
                corner_id=want[e["index"]],
                att_players=sorted(onpitch.get(att, set())),
                def_players=sorted(onpitch.get(deft, set())),
                att_gk=gk.get(att), def_gk=gk.get(deft)))
    return out


def main():
    dc = pd.read_parquet(f"{OUT}/corners.parquet")
    rows = []
    for mid, g in dc.groupby("match_id"):
        rows.extend(process_match(mid, g))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/onpitch.parquet")
    n11 = (df.att_players.str.len()
           .between(7, 11) & df.def_players.str.len().between(7, 11)).mean()
    print(f"corners: {len(df)}  sane squad sizes: {n11:.2%}  "
          f"def_gk known: {df.def_gk.notna().mean():.2%}")


if __name__ == "__main__":
    main()

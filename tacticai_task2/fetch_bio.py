"""Fetch player height/weight from Wikidata for all players in the 360 matches.

Phase A: collect distinct players (id, name, nickname) from StatsBomb lineups.
Phase B: wbsearchentities per player (nickname first, then full name),
         keeping the best candidate whose description mentions football/soccer.
Phase C: wbgetentities in batches of 50 for P2048 (height) / P2067 (mass).
Output: player_bio.parquet
"""
import json
import os
import re
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

DATA = "/Users/daniel/Desktop/open-data-master/data"
OUT = os.path.dirname(os.path.abspath(__file__))
API = "https://www.wikidata.org/w/api.php"
HDRS = {"User-Agent": "corner-kick-research/0.1 (danielytq24@gmail.com)"}
FOOT = re.compile(r"footbal|soccer", re.I)

session = requests.Session()
session.headers.update(HDRS)


def collect_players():
    players = {}
    for f in os.listdir(f"{DATA}/three-sixty"):
        if not f.endswith(".json"):
            continue
        mid = f[:-5]
        try:
            lineups = json.load(open(f"{DATA}/lineups/{mid}.json"))
        except FileNotFoundError:
            continue
        for team in lineups:
            for p in team["lineup"]:
                players[p["player_id"]] = dict(
                    player_id=p["player_id"], name=p["player_name"],
                    nickname=p.get("player_nickname"),
                    country=(p.get("country") or {}).get("name"))
    return list(players.values())


def search_one(p):
    for query in filter(None, [p["nickname"], p["name"]]):
        for attempt in range(3):
            try:
                r = session.get(API, params=dict(
                    action="wbsearchentities", search=query, language="en",
                    format="json", type="item", limit=5), timeout=15)
                hits = r.json().get("search", [])
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
                hits = []
        best = None
        for h in hits:
            desc = h.get("description", "") or ""
            if FOOT.search(desc):
                best = h["id"]
                break
        if best:
            return {**p, "qid": best}
    return {**p, "qid": None}


def qty_cm(claims, prop):
    for c in claims.get(prop, []):
        try:
            v = c["mainsnak"]["datavalue"]["value"]
            amt = float(v["amount"])
            unit = v["unit"].rsplit("/", 1)[-1]
            if prop == "P2048":
                return amt * 100 if unit == "Q11573" else amt  # m -> cm
            return amt  # kg
        except (KeyError, TypeError):
            continue
    return None


def clean_height(h):
    """Repair heights recorded in other units (m, feet) and drop outliers."""
    if h is None or pd.isna(h):
        return None
    if 1.2 <= h <= 2.3:
        h *= 100          # metres
    elif 4.0 <= h <= 7.5:
        h *= 30.48        # feet
    return h if 140 <= h <= 220 else None


def clean_weight(w):
    if w is None or pd.isna(w):
        return None
    return w if 40 <= w <= 120 else None


def main():
    players = collect_players()
    print(f"players to look up: {len(players)}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        found = list(ex.map(search_one, players))
    hit = [p for p in found if p["qid"]]
    print(f"qid found: {len(hit)}/{len(found)}", flush=True)

    qids = sorted({p["qid"] for p in hit})
    bio = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        for attempt in range(3):
            try:
                r = session.get(API, params=dict(
                    action="wbgetentities", ids="|".join(batch),
                    props="claims", format="json"), timeout=30)
                ents = r.json().get("entities", {})
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
                ents = {}
        for qid, ent in ents.items():
            claims = ent.get("claims", {})
            bio[qid] = (qty_cm(claims, "P2048"), qty_cm(claims, "P2067"))
        time.sleep(0.3)
        if i % 500 == 0:
            print(f"claims fetched: {i + len(batch)}/{len(qids)}", flush=True)

    rows = []
    for p in found:
        h, w = bio.get(p["qid"], (None, None)) if p["qid"] else (None, None)
        rows.append({**p, "height_cm": h, "weight_kg": w})
    df = pd.DataFrame(rows)
    df["height_cm"] = df.height_cm.map(clean_height)
    df["weight_kg"] = df.weight_kg.map(clean_weight)
    df.to_parquet(f"{OUT}/player_bio.parquet")
    print(f"height coverage: {df.height_cm.notna().mean():.2%}  "
          f"weight coverage: {df.weight_kg.notna().mean():.2%}")
    print(df.height_cm.describe().round(1))


if __name__ == "__main__":
    main()

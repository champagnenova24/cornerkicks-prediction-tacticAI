"""Build match_index.csv at the root of the open-data checkout.

The index is not part of the StatsBomb repository; stage1_matches.py expects
it with one row per match and the columns the code actually reads:
match_id, competition, season, gender, has_360.

  competition/season come from data/matches/<competition_id>/<season_id>.json
  gender comes from data/competitions.json (competition_gender)
  has_360 means data/three-sixty/<match_id>.json exists

274 event files (mostly Bundesliga 2015/2016) have no row in the matches
metadata; they are kept with competition '(metadata missing)' so the corpus
still covers every event file. None of them carry 360 data.

Output: <open-data>/match_index.csv
"""
import json
import os

import pandas as pd

from common import OPEN_DATA, EVENTS_DIR, THREESIXTY_DIR, MATCH_INDEX


def main():
    with open(os.path.join(OPEN_DATA, 'data', 'competitions.json')) as f:
        comps = json.load(f)
    gender = {(c['competition_id'], c['season_id']): c['competition_gender']
              for c in comps}

    has_360 = {int(f[:-5]) for f in os.listdir(THREESIXTY_DIR)
               if f.endswith('.json')}

    rows = []
    matches_dir = os.path.join(OPEN_DATA, 'data', 'matches')
    for comp_dir in sorted(os.listdir(matches_dir)):
        comp_path = os.path.join(matches_dir, comp_dir)
        if not os.path.isdir(comp_path):
            continue
        for fname in sorted(os.listdir(comp_path)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(comp_path, fname)) as f:
                for m in json.load(f):
                    comp, season = m['competition'], m['season']
                    rows.append(dict(
                        match_id=m['match_id'],
                        competition=comp['competition_name'],
                        season=season['season_name'],
                        gender=gender[(comp['competition_id'],
                                       season['season_id'])],
                        has_360=m['match_id'] in has_360))

    indexed = {r['match_id'] for r in rows}
    orphans = {int(f[:-5]) for f in os.listdir(EVENTS_DIR)
               if f.endswith('.json')} - indexed
    for mid in sorted(orphans):
        rows.append(dict(match_id=mid, competition='(metadata missing)',
                         season='unknown', gender='unknown',
                         has_360=mid in has_360))

    df = pd.DataFrame(rows).sort_values('match_id')
    df.to_csv(MATCH_INDEX, index=False)
    print(f'wrote {MATCH_INDEX}: {len(df)} matches, '
          f'{int(df.has_360.sum())} with 360')


if __name__ == '__main__':
    main()

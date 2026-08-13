"""Stage 1: match table with corpus assignment and train/val/test splits.

Reads match_index.csv, marks the 360 causal core vs the pretraining corpus,
holds out the Women's World Cup 2023 as an OOD test competition, and assigns
a deterministic by-match 80/10/10 split to everything else.

Output: data/processed/matches.parquet
"""
import pandas as pd

from common import MATCH_INDEX, PROCESSED, OOD_COMPETITION, match_split


def main():
    df = pd.read_csv(MATCH_INDEX)
    df['has_360'] = df['has_360'].astype(str).str.lower().isin(['true', '1', 'yes'])
    df['corpus'] = df['has_360'].map({True: 'causal_core', False: 'pretrain'})

    ood = (df['competition'] == OOD_COMPETITION[0]) & (df['season'] == OOD_COMPETITION[1])
    df['split'] = df['match_id'].map(match_split)
    df.loc[ood, 'split'] = 'ood_test'

    out = f'{PROCESSED}/matches.parquet'
    df.to_parquet(out, index=False)

    print(f'wrote {out}: {len(df)} matches')
    print(df.groupby(['corpus', 'split']).size().unstack(fill_value=0))


if __name__ == '__main__':
    main()

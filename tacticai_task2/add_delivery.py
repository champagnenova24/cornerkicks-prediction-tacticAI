"""Pull delivery attributes off the corner pass event for every extracted
corner: technique (Inswinging/Outswinging/Straight), height (Ground/Low/High),
body_part (Left/Right Foot), plus pass length.

Output: corner_delivery.parquet, keyed by corner_id (= the corner event id).
"""
import json
import os
import pandas as pd

DATA = "/Users/daniel/Desktop/open-data-master/data"
SP = os.path.dirname(os.path.abspath(__file__))


def main():
    dc = pd.read_parquet(f"{SP}/corners.parquet")
    want = set(dc.corner_id)
    by_match = dc.groupby("match_id").corner_id.apply(set).to_dict()

    rows = []
    for k, (mid, cids) in enumerate(by_match.items()):
        events = json.load(open(f"{DATA}/events/{mid}.json"))
        for e in events:
            if e["id"] not in cids:
                continue
            p = e.get("pass", {})
            rows.append(dict(
                corner_id=e["id"],
                technique=(p.get("technique") or {}).get("name", "Missing"),
                height=(p.get("height") or {}).get("name", "Missing"),
                body_part=(p.get("body_part") or {}).get("name", "Missing"),
                pass_length=p.get("length", float("nan")),
            ))
        if k % 100 == 0:
            print(f"{k}/{len(by_match)} matches", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(f"{SP}/corner_delivery.parquet")
    print(f"\ncorners matched: {len(df)}/{len(want)}")
    for col in ("technique", "height", "body_part"):
        print(f"\n{col}:\n{df[col].value_counts().to_string()}")
    m = dc.merge(df, on="corner_id")
    print("\nshot rate by technique:")
    print(m.groupby("technique").shot.agg(["mean", "size"]).round(3).to_string())
    print("\nshot rate by height:")
    print(m.groupby("height").shot.agg(["mean", "size"]).round(3).to_string())
    print("\nshot rate by body_part:")
    print(m.groupby("body_part").shot.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()

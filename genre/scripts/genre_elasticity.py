import pandas as pd
import numpy as np
import json
import os
import sys

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/new_tmdb_movies_master.jsonl"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/genre_elasticity.csv")
PRE_COVID_START = "2016-01-01"
PRE_COVID_END = "2019-12-31"
POST_COVID_START = "2021-01-01"
POST_COVID_END = "2024-12-31"


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    data_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)

                    data["revenue"] = float(data.get("revenue", 0))
                    data_list.append(data)
                except:
                    continue
    df = pd.DataFrame(data_list)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def calculate_elasticity():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = load_data(INPUT_FILE)

    df = df.dropna(subset=["release_date"])
    df = df[(df["vote_count"] >= 10) & (df["revenue"] > 0)]

    pre_covid = df[
        (df["release_date"] >= PRE_COVID_START) & (df["release_date"] <= PRE_COVID_END)
    ].copy()
    post_covid = df[
        (df["release_date"] >= POST_COVID_START)
        & (df["release_date"] <= POST_COVID_END)
    ].copy()

    R_total_pre = pre_covid["revenue"].sum()
    N_total_pre = len(pre_covid)

    R_total_post = post_covid["revenue"].sum()
    N_total_post = len(post_covid)

    print(f"Pre-COVID (2016-2019): N={N_total_pre}, Revenue=${R_total_pre:,.0f}")
    print(f"Post-COVID (2021-2024): N={N_total_post}, Revenue=${R_total_post:,.0f}")

    if N_total_pre == 0 or N_total_post == 0:
        print("Error: No movies in one of the periods.")
        return

    def aggregate_by_genre(dataframe):
        genre_stats = {}
        for _, row in dataframe.iterrows():
            genres = row.get("genres", [])
            rev = row.get("revenue", 0)
            if isinstance(genres, list):
                for g in genres:
                    g_name = (
                        g["name"]
                        if isinstance(g, dict) and "name" in g
                        else g if isinstance(g, str) else None
                    )
                    if g_name:
                        if g_name not in genre_stats:
                            genre_stats[g_name] = {"count": 0, "revenue": 0.0}
                        genre_stats[g_name]["count"] += 1
                        genre_stats[g_name]["revenue"] += rev
        return genre_stats

    stats_pre = aggregate_by_genre(pre_covid)
    stats_post = aggregate_by_genre(post_covid)

    all_genres = sorted(list(set(stats_pre.keys()) | set(stats_post.keys())))

    results = []

    for genre in all_genres:

        pre = stats_pre.get(genre, {"count": 0, "revenue": 0})
        rev_share_pre = pre["revenue"] / R_total_pre if R_total_pre > 0 else 0
        vol_share_pre = pre["count"] / N_total_pre

        post = stats_post.get(genre, {"count": 0, "revenue": 0})
        rev_share_post = post["revenue"] / R_total_post if R_total_post > 0 else 0
        vol_share_post = post["count"] / N_total_post

        if rev_share_pre > 0.001:
            pct_delta_rev = (rev_share_post - rev_share_pre) / rev_share_pre
        else:
            pct_delta_rev = None

        if vol_share_pre > 0.001:
            pct_delta_vol = (vol_share_post - vol_share_pre) / vol_share_pre
        else:
            pct_delta_vol = None

        if (
            pct_delta_vol is not None
            and pct_delta_rev is not None
            and pct_delta_vol != 0
        ):
            elasticity = pct_delta_rev / pct_delta_vol
        else:
            elasticity = None

        results.append(
            {
                "Genre": genre,
                "Pre_Vol_Share": vol_share_pre,
                "Post_Vol_Share": vol_share_post,
                "Pct_Delta_Vol": pct_delta_vol,
                "Pre_Rev_Share": rev_share_pre,
                "Post_Rev_Share": rev_share_post,
                "Pct_Delta_Rev": pct_delta_rev,
                "Elasticity": elasticity,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Pre_Vol_Share", ascending=False)

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\nElasticity Results (Sorted by Pre-Covid Volume Share):")
    cols = [
        "Genre",
        "Pre_Vol_Share",
        "Pct_Delta_Vol",
        "Pre_Rev_Share",
        "Pct_Delta_Rev",
        "Elasticity",
    ]
    print(results_df[cols].head(15).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    calculate_elasticity()

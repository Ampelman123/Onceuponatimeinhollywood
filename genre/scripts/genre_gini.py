import pandas as pd
import numpy as np
import json
import os
import sys

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/new_tmdb_movies_master.jsonl"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/genre_gini.csv")
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
                    data["vote_count"] = int(data.get("vote_count", 0))
                    data_list.append(data)
                except:
                    continue
    df = pd.DataFrame(data_list)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def gini(array):
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        return -1

    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    if n == 0 or np.sum(array) == 0:
        return 0

    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_gini_coefficients():
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

    print(f"Pre-COVID Analysis Set: {len(pre_covid)} movies")
    print(f"Post-COVID Analysis Set: {len(post_covid)} movies")

    def analyze_period(dataframe, period_name):
        genre_revenues = {}
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
                        if g_name not in genre_revenues:
                            genre_revenues[g_name] = []
                        genre_revenues[g_name].append(rev)

        results = {}
        for genre, revenues in genre_revenues.items():
            if len(revenues) > 5:
                results[genre] = gini(revenues)
        return results

    gini_pre = analyze_period(pre_covid, "Pre-COVID")
    gini_post = analyze_period(post_covid, "Post-COVID")

    all_genres = sorted(list(set(gini_pre.keys()) | set(gini_post.keys())))

    final_results = []
    for genre in all_genres:
        g_pre = gini_pre.get(genre, np.nan)
        g_post = gini_post.get(genre, np.nan)

        delta = (
            g_post - g_pre if not np.isnan(g_pre) and not np.isnan(g_post) else np.nan
        )

        final_results.append(
            {
                "Genre": genre,
                "Pre_Gini": g_pre,
                "Post_Gini": g_post,
                "Delta_Gini": delta,
            }
        )

    results_df = pd.DataFrame(final_results)
    results_df = results_df.dropna(subset=["Delta_Gini"])
    results_df = results_df.sort_values(by="Delta_Gini", ascending=False)

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\nTop Gini Increases (Becoming More Unequal):")
    print(results_df.head(5).to_string(index=False))

    print("\nTop Gini Decreases (Becoming More Equal):")
    print(results_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    calculate_gini_coefficients()

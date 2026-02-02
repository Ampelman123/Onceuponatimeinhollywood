import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style, COLORS

set_style(column="full")

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/new_tmdb_movies_master.jsonl"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "efficiency_fragility_matrix.pdf")
PRE_COVID_YEARS = [2016, 2017, 2018, 2019]
POST_COVID_YEARS = [2021, 2022, 2023, 2024]
GINI_FILE = os.path.join(os.path.dirname(__file__), "../../data/genre_gini.csv")


def load_data():
    data_list = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
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
    df["year"] = df["release_date"].dt.year
    df = df[(df["vote_count"] >= 10) & (df["revenue"] > 0)].copy()

    return df


def calculate_metrics():
    df = load_data()

    def get_period(year):
        if year in PRE_COVID_YEARS:
            return "Pre-COVID"
        if year in POST_COVID_YEARS:
            return "Post-COVID"
        return None

    df["period"] = df["year"].apply(get_period)
    df = df.dropna(subset=["period"])

    stats_list = []

    for period in ["Pre-COVID", "Post-COVID"]:
        sub = df[df["period"] == period]
        total_rev = sub["revenue"].sum()
        total_count = len(sub)

        genre_rows = []
        for _, row in sub.iterrows():
            if isinstance(row["genres"], list):
                for g in row["genres"]:
                    g_name = (
                        g["name"]
                        if isinstance(g, dict) and "name" in g
                        else g if isinstance(g, str) else None
                    )
                    if g_name:
                        genre_rows.append({"genre": g_name, "revenue": row["revenue"]})

        g_df = pd.DataFrame(genre_rows)
        vol_counts = g_df["genre"].value_counts(normalize=True)
        rev_sums = g_df.groupby("genre")["revenue"].sum()
        rev_share = rev_sums / total_rev

        combined = pd.DataFrame({"vol_share": vol_counts, "rev_share": rev_share})
        combined["efficiency_gap"] = combined["rev_share"] - combined["vol_share"]
        combined["period"] = period
        combined["count"] = g_df["genre"].value_counts()
        stats_list.append(combined)

    pre = stats_list[0]
    post = stats_list[1]
    genres = sorted(list(set(pre.index) | set(post.index)))

    metrics = []
    for g in genres:
        eff_pre = pre.loc[g, "efficiency_gap"] if g in pre.index else 0
        eff_post = post.loc[g, "efficiency_gap"] if g in post.index else 0
        delta_eff = eff_post - eff_pre

        post_count = post.loc[g, "count"] if g in post.index else 0

        metrics.append(
            {
                "Genre": g,
                "Delta_Efficiency": delta_eff * 100,
                "Post_Volume": post_count,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    gini_df = pd.read_csv(GINI_FILE)
    final_df = pd.merge(
        metrics_df, gini_df[["Genre", "Post_Gini"]], on="Genre", how="inner"
    )

    return final_df


def plot_matrix():
    df = calculate_metrics()

    fig, ax = plt.subplots()
    sizes = df["Post_Volume"] * 2

    ax.scatter(
        df["Delta_Efficiency"],
        df["Post_Gini"],
        s=sizes,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        color=COLORS["revenue"],
    )

    ax.axvline(0, color=COLORS["text_light"], linestyle="--", alpha=0.5)
    median_gini = df["Post_Gini"].median()
    ax.axhline(median_gini, color=COLORS["text_light"], linestyle="--", alpha=0.5)

    for idx, row in df.iterrows():
        is_large = row["Post_Volume"] > df["Post_Volume"].quantile(0.6)
        is_extreme_x = abs(row["Delta_Efficiency"]) > 1.5
        is_extreme_y = abs(row["Post_Gini"] - median_gini) > 0.05

        if is_large or is_extreme_x or is_extreme_y:
            ax.text(
                row["Delta_Efficiency"],
                row["Post_Gini"],
                row["Genre"],
                ha="center",
                va="center",
            )

    ax.set_xlabel("Change in Efficiency Gap (Pre vs Post) [pp]")
    ax.set_ylabel("Fragility (Post-COVID Gini Coefficient)")

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_matrix()

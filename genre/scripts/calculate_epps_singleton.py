import pandas as pd
import numpy as np
import scipy.stats as stats
import ast
import warnings
import os

try:
    from scipy.stats import epps_singleton_2samp
except ImportError:
    print("Error: scipy.stats.epps_singleton_2samp not found. Please upgrade scipy.")
    exit(1)

warnings.simplefilter(action="ignore", category=FutureWarning)

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/dataset_final.csv")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/epps_singleton_metrics.csv"
)
PRE_COVID_YEARS = [2016, 2017, 2018, 2019]
POST_COVID_YEARS = [2021, 2022, 2023, 2024]


def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    def safe_parse(val):
        if isinstance(val, str) and val.strip():
            try:
                return ast.literal_eval(val)
            except:
                return val
        return val

    df["genres"] = df["genres"].apply(safe_parse)
    df["belongs_to_collection"] = df["belongs_to_collection"].apply(safe_parse)

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    df["is_franchise"] = df["belongs_to_collection"].apply(
        lambda x: 1 if pd.notnull(x) else 0
    )

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    df = df[(df["vote_count"] >= 10) & (df["revenue"] > 0)].copy()

    def get_period(year):
        if year in PRE_COVID_YEARS:
            return "Pre-COVID"
        if year in POST_COVID_YEARS:
            return "Post-COVID"
        return None

    df["period"] = df["year"].apply(get_period)
    df = df.dropna(subset=["period"]).copy()
    return df


def calculate_w_statistic(data1, data2):
    if len(data1) < 5 or len(data2) < 5:
        return np.nan, np.nan

    try:
        w_stat, p_val = epps_singleton_2samp(data1, data2)
        return w_stat, p_val
    except Exception:
        return np.nan, np.nan


def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = load_and_clean_data(INPUT_FILE)
    print(f"Total Analysis Movies: {len(df)}")

    results = []

    pre_rev = df[df["period"] == "Pre-COVID"]["revenue"]
    post_rev = df[df["period"] == "Post-COVID"]["revenue"]

    w, p = calculate_w_statistic(pre_rev, post_rev)
    results.append(
        {
            "Scope": "Overall Industry",
            "Genre": "All",
            "Type": "All Movies",
            "W_Statistic": w,
            "p_Value": p,
            "Pre_N": len(pre_rev),
            "Post_N": len(post_rev),
        }
    )

    df_exploded = df.explode("genres")
    df_exploded["genre_name"] = df_exploded["genres"].apply(
        lambda x: (
            x.get("name") if isinstance(x, dict) else x if isinstance(x, str) else None
        )
    )
    df_exploded = df_exploded.dropna(subset=["genre_name"])

    genres = df_exploded["genre_name"].unique()

    print(f"Calculating for {len(genres)} genres...")

    for g in sorted(genres):
        g_df = df_exploded[df_exploded["genre_name"] == g]

        pre_g = g_df[g_df["period"] == "Pre-COVID"]["revenue"]
        post_g = g_df[g_df["period"] == "Post-COVID"]["revenue"]

        w, p = calculate_w_statistic(pre_g, post_g)
        results.append(
            {
                "Scope": "Genre",
                "Genre": g,
                "Type": "All Movies",
                "W_Statistic": w,
                "p_Value": p,
                "Pre_N": len(pre_g),
                "Post_N": len(post_g),
            }
        )

        g_fran = g_df[g_df["is_franchise"] == 1]
        pre_f = g_fran[g_fran["period"] == "Pre-COVID"]["revenue"]
        post_f = g_fran[g_fran["period"] == "Post-COVID"]["revenue"]

        w_f, p_f = calculate_w_statistic(pre_f, post_f)
        results.append(
            {
                "Scope": "Genre",
                "Genre": g,
                "Type": "Franchise Only",
                "W_Statistic": w_f,
                "p_Value": p_f,
                "Pre_N": len(pre_f),
                "Post_N": len(post_f),
            }
        )

    res_df = pd.DataFrame(results)
    res_df = res_df.dropna(subset=["W_Statistic"])

    print("\nTop 5 Significant Changes (All Movies):")
    print(
        res_df[res_df["Type"] == "All Movies"]
        .sort_values(by="W_Statistic", ascending=False)
        .head(5)
    )

    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_analysis()

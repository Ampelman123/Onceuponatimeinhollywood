import pandas as pd
import numpy as np
import scipy.stats as stats
import ast
import warnings
import os
import sys

warnings.simplefilter(action="ignore", category=FutureWarning)

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/dataset_final.csv")
PRE_COVID_YEARS = [2016, 2017, 2018, 2019]
POST_COVID_YEARS = [2021, 2022, 2023, 2024]
N_BOOTSTRAP = (
    1000  # Default to 1000 for script execution, notebook had 10000 but that's slow
)


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
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)

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


def compute_metrics(dataframe, genres):
    period_stats = {}
    for period in ["Pre-COVID", "Post-COVID"]:
        sub = dataframe[dataframe["period"] == period]
        total_rev = sub["revenue"].sum()
        total_budget = sub["budget"].sum()
        total_count = len(sub)

        g_rows = []
        for _, row in sub.iterrows():
            if isinstance(row["genres"], list):
                for g in row["genres"]:
                    g_name = (
                        g["name"]
                        if isinstance(g, dict) and "name" in g
                        else g if isinstance(g, str) else None
                    )
                    if g_name and g_name in genres:
                        g_rows.append(
                            {
                                "genre": g_name,
                                "revenue": row.get("revenue", 0),
                                "budget": row.get("budget", 0),
                                "is_franchise": row.get("is_franchise", 0),
                            }
                        )
        g_df = pd.DataFrame(g_rows)

        if not g_df.empty:
            vol_counts = g_df["genre"].value_counts()
            budget_sums = g_df.groupby("genre")["budget"].sum()
            rev_sums = g_df.groupby("genre")["revenue"].sum()
            fran_rev_sums = (
                g_df[g_df["is_franchise"] == 1].groupby("genre")["revenue"].sum()
            )
            rev_arrays = g_df.groupby("genre")["revenue"].apply(list)
            budg_arrays = g_df.groupby("genre")["budget"].apply(list)
        else:
            vol_counts = pd.Series(dtype=float)
            budget_sums = pd.Series(dtype=float)
            rev_sums = pd.Series(dtype=float)
            fran_rev_sums = pd.Series(dtype=float)
            rev_arrays = pd.Series(dtype=object)
            budg_arrays = pd.Series(dtype=object)

        stats = {}
        for g in genres:
            count = vol_counts.get(g, 0)
            budg = budget_sums.get(g, 0)
            rev = rev_sums.get(g, 0)

            vol_share = count / total_count if total_count > 0 else 0
            budg_share = budg / total_budget if total_budget > 0 else 0
            rev_share = rev / total_rev if total_rev > 0 else 0

            fran_rev = fran_rev_sums.get(g, 0)
            fran_rev_share = fran_rev / rev if rev > 0 else 0

            g_gini = 0
            if g in rev_arrays.index and len(rev_arrays[g]) > 1:
                g_gini = gini(rev_arrays[g])

            g_gini_budget = 0
            if g in budg_arrays.index and len(budg_arrays[g]) > 1:
                g_gini_budget = gini(budg_arrays[g])

            stats[g] = {
                "count": count,
                "vol_share": vol_share,
                "rev_share": rev_share,
                "budg_share": budg_share,
                "fran_rev_share": fran_rev_share,
                "gini": g_gini,
                "gini_budget": g_gini_budget,
            }
        period_stats[period] = stats, total_count

    return period_stats


def calculate_shifts(period_stats, genres):
    pre_stats, n1 = period_stats["Pre-COVID"]
    post_stats, n2 = period_stats["Post-COVID"]

    results = {}
    for g in genres:
        s1 = pre_stats[g]
        s2 = post_stats[g]

        d_vol = (
            (s2["vol_share"] - s1["vol_share"]) / s1["vol_share"]
            if s1["vol_share"] > 0
            else 0
        )
        diff_prop = s2["vol_share"] - s1["vol_share"]

        d_rev = (
            (s2["rev_share"] - s1["rev_share"]) / s1["rev_share"]
            if s1["rev_share"] > 0
            else 0
        )
        diff_rev_share = s2["rev_share"] - s1["rev_share"]

        d_budg = (
            (s2["budg_share"] - s1["budg_share"]) / s1["budg_share"]
            if s1["budg_share"] > 0
            else 0
        )
        diff_budg_share = s2["budg_share"] - s1["budg_share"]

        elast_price = d_rev / d_vol if d_vol != 0 else 0
        elast_effic = d_rev / d_budg if d_budg != 0 else 0

        d_gini = s2["gini"] - s1["gini"]
        d_gini_budget = s2["gini_budget"] - s1["gini_budget"]

        diff_fran_rev_share = s2["fran_rev_share"] - s1["fran_rev_share"]

        results[g] = {
            "diff_prop": diff_prop,
            "d_rev": d_rev,
            "diff_rev_share": diff_rev_share,
            "d_budg": d_budg,
            "diff_budg_share": diff_budg_share,
            "elasticity": elast_price,
            "elasticity_efficiency": elast_effic,
            "d_gini": d_gini,
            "d_gini_budget": d_gini_budget,
            "diff_fran_rev_share": diff_fran_rev_share,
            "p1": s1["vol_share"],
            "p2": s2["vol_share"],
            "x1": s1["count"],
            "x2": s2["count"],
        }
    return results, n1, n2


def run_bootstrap_analysis(dataframe, label="General"):
    dataframe = dataframe.dropna(subset=["period"]).copy()
    print(f"Running Analysis for: {label} (Movies: {len(dataframe)})...")

    all_genres = set()
    for row in dataframe["genres"]:
        if isinstance(row, list):
            for g in row:
                if isinstance(g, dict):
                    all_genres.add(g["name"])
    genres = sorted(list(all_genres))

    obs_period_stats = compute_metrics(dataframe, genres)
    obs_results, n1, n2 = calculate_shifts(obs_period_stats, genres)

    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
    bootstrap_vals = {
        g: {
            "diff_prop": [],
            "elasticity": [],
            "d_gini": [],
            "diff_budg": [],
            "effic_elast": [],
            "d_gini_budg": [],
            "diff_rev": [],
            "diff_fran_rev": [],
        }
        for g in genres
    }

    pre_df = dataframe[dataframe["period"] == "Pre-COVID"]
    post_df = dataframe[dataframe["period"] == "Post-COVID"]

    for i in range(N_BOOTSTRAP):
        res_pre = pre_df.sample(frac=1, replace=True)
        res_post = post_df.sample(frac=1, replace=True)
        res_df = pd.concat([res_pre, res_post])

        iter_stats = compute_metrics(res_df, genres)
        iter_results, _, _ = calculate_shifts(iter_stats, genres)

        for g in genres:
            r = iter_results[g]
            bootstrap_vals[g]["diff_prop"].append(r["diff_prop"])
            bootstrap_vals[g]["elasticity"].append(r["elasticity"])
            bootstrap_vals[g]["d_gini"].append(r["d_gini"])
            bootstrap_vals[g]["diff_budg"].append(r["diff_budg_share"])
            bootstrap_vals[g]["effic_elast"].append(r["elasticity_efficiency"])
            bootstrap_vals[g]["d_gini_budg"].append(r["d_gini_budget"])
            bootstrap_vals[g]["diff_rev"].append(r["diff_rev_share"])
            bootstrap_vals[g]["diff_fran_rev"].append(r["diff_fran_rev_share"])

    final_table = []
    for g in genres:
        obs = obs_results[g]

        p_pool = (obs["x1"] + obs["x2"]) / (n1 + n2)
        se_vol = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_vol = (obs["p2"] - obs["p1"]) / se_vol if se_vol > 0 else 0
        p_z_vol = 2 * (1 - stats.norm.cdf(abs(z_vol)))

        se_budg = np.std(bootstrap_vals[g]["diff_budg"], ddof=1)
        z_budg = obs["diff_budg_share"] / se_budg if se_budg > 0 else 0
        p_z_budg = 2 * (1 - stats.norm.cdf(abs(z_budg)))

        se_rev = np.std(bootstrap_vals[g]["diff_rev"], ddof=1)
        z_rev = obs["diff_rev_share"] / se_rev if se_rev > 0 else 0
        p_z_rev = 2 * (1 - stats.norm.cdf(abs(z_rev)))

        se_fran_rev = np.std(bootstrap_vals[g]["diff_fran_rev"], ddof=1)
        z_fran_rev = obs["diff_fran_rev_share"] / se_fran_rev if se_fran_rev > 0 else 0
        p_z_fran_rev = 2 * (1 - stats.norm.cdf(abs(z_fran_rev)))

        dist_elast = np.array(bootstrap_vals[g]["elasticity"])
        p_elast = 2 * min(np.mean(dist_elast > 0), np.mean(dist_elast < 0))

        dist_effic = np.array(bootstrap_vals[g]["effic_elast"])
        p_effic = 2 * min(np.mean(dist_effic > 0), np.mean(dist_effic < 0))

        dist_gini = np.array(bootstrap_vals[g]["d_gini"])
        p_gini = 2 * min(np.mean(dist_gini > 0), np.mean(dist_gini < 0))

        dist_gini_budg = np.array(bootstrap_vals[g]["d_gini_budg"])
        p_gini_budg = 2 * min(np.mean(dist_gini_budg > 0), np.mean(dist_gini_budg < 0))

        final_table.append(
            {
                "Genre": g,
                "Z_Vol": z_vol,
                "p_Vol": p_z_vol,
                "Z_Budget": z_budg,
                "p_Budget": p_z_budg,
                "Z_Revenue": z_rev,
                "p_Revenue": p_z_rev,
                "Z_Franchise_Saturation": z_fran_rev,
                "p_Franchise_Saturation": p_z_fran_rev,
                "Elasticity": obs["elasticity"],
                "p_Elasticity": p_elast,
                "Efficiency": obs["elasticity_efficiency"],
                "p_Efficiency": p_effic,
                "Delta_Gini": obs["d_gini"],
                "p_Gini": p_gini,
                "Delta_Budget_Gini": obs["d_gini_budget"],
                "p_Budget_Gini": p_gini_budg,
            }
        )

    return pd.DataFrame(final_table)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = load_and_clean_data(INPUT_FILE)
    print(f"Total Movies: {len(df)}")
    print(f"Franchise Movies: {len(df[df['is_franchise']==1])}")

    # General Analysis
    res_general = run_bootstrap_analysis(df, "General")
    print("\nGeneral Analysis Results (Top 5 Volume Shifts):")
    print(res_general.sort_values(by="Z_Vol", key=abs, ascending=False).head(5))

    # Export General Results
    file_gen_metrics = os.path.join(
        os.path.dirname(__file__), "../../data/genre_metrics_compiled.csv"
    )
    res_general[
        [
            "Genre",
            "Z_Vol",
            "p_Vol",
            "Z_Revenue",
            "p_Revenue",
            "Z_Franchise_Saturation",
            "p_Franchise_Saturation",
            "Elasticity",
            "p_Elasticity",
            "Delta_Gini",
            "p_Gini",
        ]
    ].rename(
        columns={
            "Z_Vol": "Z_Score",
            "p_Vol": "p_Value",
            "Z_Revenue": "Z_Score_Revenue",
            "p_Revenue": "p_Value_Revenue",
            "Delta_Gini": "Delta_Gini",
        }
    ).to_csv(
        file_gen_metrics, index=False
    )
    print(f"Saved {file_gen_metrics}")

    file_gen_budget = os.path.join(
        os.path.dirname(__file__), "../../data/genre_budget_metrics.csv"
    )
    res_general[
        [
            "Genre",
            "Z_Budget",
            "p_Budget",
            "Efficiency",
            "p_Efficiency",
            "Delta_Budget_Gini",
            "p_Budget_Gini",
        ]
    ].rename(
        columns={
            "Z_Budget": "Z_Score_Budget",
            "p_Budget": "p_Z",
            "Efficiency": "Elasticity_Efficiency",
            "Delta_Budget_Gini": "Delta_Budget_Gini",
            "p_Budget_Gini": "p_Gini",
        }
    ).to_csv(
        file_gen_budget, index=False
    )
    print(f"Saved {file_gen_budget}")

    # Franchise Analysis
    df_franchise = df[df["is_franchise"] == 1].copy()
    res_franchise = run_bootstrap_analysis(df_franchise, "Franchise")
    print("\nFranchise Analysis Results (Top 5 Volume Shifts):")
    print(res_franchise.sort_values(by="Z_Vol", key=abs, ascending=False).head(5))

    # Export Franchise Results
    file_fran_metrics = os.path.join(
        os.path.dirname(__file__), "../../data/franchise_metrics_compiled.csv"
    )
    res_franchise[
        [
            "Genre",
            "Z_Vol",
            "p_Vol",
            "Z_Revenue",
            "p_Revenue",
            "Z_Franchise_Saturation",
            "p_Franchise_Saturation",
            "Elasticity",
            "p_Elasticity",
            "Delta_Gini",
            "p_Gini",
        ]
    ].rename(
        columns={
            "Z_Vol": "Z_Score",
            "p_Vol": "p_Value",
            "Z_Revenue": "Z_Score_Revenue",
            "p_Revenue": "p_Value_Revenue",
        }
    ).to_csv(
        file_fran_metrics, index=False
    )
    print(f"Saved {file_fran_metrics}")

    file_fran_budget = os.path.join(
        os.path.dirname(__file__), "../../data/franchise_budget_metrics.csv"
    )
    res_franchise[
        [
            "Genre",
            "Z_Budget",
            "p_Budget",
            "Efficiency",
            "p_Efficiency",
            "Delta_Budget_Gini",
            "p_Budget_Gini",
        ]
    ].rename(
        columns={
            "Z_Budget": "Z_Score_Budget",
            "p_Budget": "p_Z",
            "Efficiency": "Elasticity_Efficiency",
            "Delta_Budget_Gini": "Delta_Budget_Gini",
            "p_Budget_Gini": "p_Gini",
        }
    ).to_csv(
        file_fran_budget, index=False
    )
    print(f"Saved {file_fran_budget}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import scipy.stats as stats
import ast
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/dataset_final.csv")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/overall_industry_metrics.csv"
)
PRE_COVID_YEARS = [2016, 2017, 2018, 2019]
POST_COVID_YEARS = [2021, 2022, 2023, 2024]
N_BOOTSTRAP = 100


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


df = load_and_clean_data(INPUT_FILE)
print(f"Total Movies: {len(df)}")


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


def calculate_overall_metrics(dataframe):
    stats_dict = {}
    for period in ["Pre-COVID", "Post-COVID"]:
        sub = dataframe[dataframe["period"] == period]
        total_n = len(sub)
        if total_n == 0:
            continue

        mean_rev = sub["revenue"].mean()
        mean_bud = sub["budget"].mean()

        rev_list = sub["revenue"].tolist()
        bud_list = sub["budget"].tolist()
        gini_rev = gini(rev_list) if len(rev_list) > 0 else 0
        gini_bud = gini(bud_list) if len(bud_list) > 0 else 0

        exploded = sub.explode("genres")
        filtered_genres = exploded[
            exploded["genres"].apply(lambda x: isinstance(x, dict))
        ]
        if not filtered_genres.empty:
            filtered_genres["genre_name"] = filtered_genres["genres"].apply(
                lambda x: x.get("name")
            )
            g_stats = (
                filtered_genres.groupby("genre_name")
                .agg({"revenue": "sum", "budget": "sum", "title": "count"})
                .rename(columns={"title": "count"})
            )

            gini_genre_vol = gini(g_stats["count"].values) if len(g_stats) > 0 else 0
            gini_genre_rev = gini(g_stats["revenue"].values) if len(g_stats) > 0 else 0
            gini_genre_bud = gini(g_stats["budget"].values) if len(g_stats) > 0 else 0
        else:
            gini_genre_vol, gini_genre_rev, gini_genre_bud = 0, 0, 0

        fran_sub = sub[sub["is_franchise"] == 1]
        fran_n = len(fran_sub)
        fran_vol_share = fran_n / total_n

        total_rev = sub["revenue"].sum()
        fran_rev = fran_sub["revenue"].sum()
        fran_rev_share = fran_rev / total_rev if total_rev > 0 else 0

        total_bud = sub["budget"].sum()
        fran_bud = fran_sub["budget"].sum()
        fran_bud_share = fran_bud / total_bud if total_bud > 0 else 0

        stats_dict[period] = {
            "Mean_Revenue": mean_rev,
            "Mean_Budget": mean_bud,
            "Overall_Gini_Coefficient": gini_rev,
            "Overall_Bud_Gini": gini_bud,
            "Genre_Vol_Gini": gini_genre_vol,
            "Genre_Rev_Gini": gini_genre_rev,
            "Genre_Bud_Gini": gini_genre_bud,
            "Franchise_Vol_Share": fran_vol_share,
            "Franchise_Rev_Share": fran_rev_share,
            "Franchise_Bud_Share": fran_bud_share,
            "Total_Volume": total_n,
        }
    return stats_dict


print("Running Overall & Franchise Shift Analysis...")
obs_overall = calculate_overall_metrics(df)

boot_diffs = {
    "Mean_Revenue": [],
    "Mean_Budget": [],
    "Overall_Gini_Coefficient": [],
    "Overall_Bud_Gini": [],
    "Genre_Vol_Gini": [],
    "Genre_Rev_Gini": [],
    "Genre_Bud_Gini": [],
    "Franchise_Vol_Share": [],
    "Franchise_Rev_Share": [],
    "Franchise_Bud_Share": [],
}

pre_df = df[df["period"] == "Pre-COVID"]
post_df = df[df["period"] == "Post-COVID"]

print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
for i in range(N_BOOTSTRAP):
    if i % 1000 == 0:
        print(f"Iteration {i}...")
    res_pre = pre_df.sample(frac=1, replace=True)
    res_post = post_df.sample(frac=1, replace=True)
    res_df = pd.concat([res_pre, res_post])

    iter_stats = calculate_overall_metrics(res_df)
    for metric in boot_diffs.keys():
        diff = iter_stats["Post-COVID"][metric] - iter_stats["Pre-COVID"][metric]
        boot_diffs[metric].append(diff)

overall_results = []
for metric in boot_diffs.keys():
    obs_diff = obs_overall["Post-COVID"][metric] - obs_overall["Pre-COVID"][metric]
    se = np.std(boot_diffs[metric], ddof=1)
    z_score = obs_diff / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    overall_results.append(
        {
            "Metric": metric,
            "Pre_Value": obs_overall["Pre-COVID"][metric],
            "Post_Value": obs_overall["Post-COVID"][metric],
            "Delta": obs_diff,
            "Z_Score": z_score,
            "p_Value": p_value,
        }
    )

n1 = obs_overall["Pre-COVID"]["Total_Volume"]
n2 = obs_overall["Post-COVID"]["Total_Volume"]
delta_n = n2 - n1
se_n = np.sqrt(n1 + n2)
z_n = delta_n / se_n if se_n > 0 else 0
p_n = 2 * (1 - stats.norm.cdf(abs(z_n)))

overall_results.insert(
    0,
    {
        "Metric": "Total_Volume",
        "Pre_Value": n1,
        "Post_Value": n2,
        "Delta": delta_n,
        "Z_Score": z_n,
        "p_Value": p_n,
    },
)

res_overall_df = pd.DataFrame(overall_results)
print("Overall Industry Shifts:")
print(res_overall_df)
res_overall_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE}")

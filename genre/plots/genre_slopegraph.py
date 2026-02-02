import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style, COLORS

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/raw_dataset_arda.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "genre_slopegraph.pdf")

PRE_COVID_YEARS = [2016, 2017, 2018, 2019]
POST_COVID_YEARS = [2021, 2022, 2023, 2024]

if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)

    def safe_parse(val):
        if isinstance(val, str) and val.strip():
            try:
                return ast.literal_eval(val)
            except:
                return val
        return val

    df["genres"] = df["genres"].apply(safe_parse)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
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

genre_rows = []
for idx, row in df.iterrows():
    if not isinstance(row["genres"], list):
        continue

    for g in row["genres"]:
        genre_name = g.get("name") if isinstance(g, dict) else g
        if genre_name == "TV Movie" or not genre_name:
            continue

        genre_rows.append(
            {
                "period": row["period"],
                "genre": genre_name,
                "revenue": row.get("revenue", 0),
            }
        )

df_flat = pd.DataFrame(genre_rows)

period_grouped = df_flat.groupby(["period", "genre"])["revenue"].sum().reset_index()

period_totals = (
    df_flat.groupby("period")["revenue"]
    .sum()
    .reset_index()
    .rename(columns={"revenue": "ind_rev"})
)

merged = pd.merge(period_grouped, period_totals, on="period")
merged["rev_share"] = (merged["revenue"] / merged["ind_rev"]) * 100

top_genres = (
    df_flat.groupby("genre")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .index.tolist()
)
plot_data = merged[merged["genre"].isin(top_genres)].copy()

num_genres = len(top_genres)
cols = 4
rows = 3

set_style(column="full", nrows=rows, ncols=cols)

fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
axes = axes.flatten()

pivoted = plot_data.pivot(index="genre", columns="period", values="rev_share")
pivoted = pivoted[["Pre-COVID", "Post-COVID"]]

ymin = pivoted.min().min()
ymax = pivoted.max().max()
pad = (ymax - ymin) * 0.1

for i, genre in enumerate(top_genres):
    ax = axes[i]
    if genre not in pivoted.index:
        continue

    start_val = pivoted.loc[genre, "Pre-COVID"]
    end_val = pivoted.loc[genre, "Post-COVID"]

    color = COLORS["increase"] if end_val > start_val else COLORS["decrease"]

    ax.plot([0, 1], [start_val, end_val], color=color, linewidth=2.5, marker="o")

    ax.text(
        -0.1,
        start_val,
        f"{start_val:.1f}%",
        ha="right",
        va="center",
        color=COLORS["text_main"],
    )
    ax.text(
        1.1,
        end_val,
        f"{end_val:.1f}%",
        ha="left",
        va="center",
        color=color,
        fontweight="bold",
    )

    ax.set_title(genre, pad=10)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.grid(axis="y", linestyle=":", alpha=0.3)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.text(
    0.5,
    -0.05,
    "Figure 1: The Great Reorientation—Slopegraphs indicate the net change in Revenue Share from the Pre-COVID baseline\n"
    "to the Post-COVID Adjustment period; note the sharp decline in Action/Adventure spectacle compared to the resilience of Family/Horror genres.",
    ha="center",
    style="italic",
)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

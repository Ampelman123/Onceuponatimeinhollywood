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
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "genre_drift_plot.pdf")

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
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
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
                "budget": row.get("budget", 0),
            }
        )

df_flat = pd.DataFrame(genre_rows)

period_grouped = (
    df_flat.groupby(["period", "genre"])
    .agg(total_rev=("revenue", "sum"), total_budg=("budget", "sum"))
    .reset_index()
)

period_totals = (
    df_flat.groupby("period")
    .agg(ind_rev=("revenue", "sum"), ind_budg=("budget", "sum"))
    .reset_index()
)

merged = pd.merge(period_grouped, period_totals, on="period")

merged["rev_share"] = (merged["total_rev"] / merged["ind_rev"]) * 100
merged["budg_share"] = (merged["total_budg"] / merged["ind_budg"]) * 100

top_genres = (
    df_flat.groupby("genre")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)
plot_data = merged[merged["genre"].isin(top_genres)].copy()

pivoted = plot_data.pivot(
    index="genre", columns="period", values=["rev_share", "budg_share"]
)
pivoted.columns = [f"{c[1]}_{c[0]}" for c in pivoted.columns]

df_vect = pd.DataFrame(
    {
        "x1": pivoted["Pre-COVID_budg_share"],
        "y1": pivoted["Pre-COVID_rev_share"],
        "x2": pivoted["Post-COVID_budg_share"],
        "y2": pivoted["Post-COVID_rev_share"],
    }
).reset_index()

set_style()

fig, ax = plt.subplots(figsize=(10, 8))

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]) + 5,
]
max_val = max(df_vect[["x1", "x2", "y1", "y2"]].max().max(), 20)
ax.plot(
    [0, max_val], [0, max_val], "k--", alpha=0.2, label="Fair Return (1:1)", zorder=1
)

for idx, row in df_vect.iterrows():
    genre = row["genre"]
    color = getattr(plt.cm, "tab10")(idx % 10)

    ax.annotate(
        "",
        xy=(row["x2"], row["y2"]),
        xycoords="data",
        xytext=(row["x1"], row["y1"]),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=0, shrinkB=0),
        zorder=2,
    )

    ax.scatter(row["x1"], row["y1"], color=color, s=50, alpha=0.6, zorder=3)

    ax.scatter(
        row["x2"], row["y2"], color=color, s=150, alpha=0.9, zorder=4, label=genre
    )

    ax.text(
        row["x2"],
        row["y2"] + 0.5,
        genre,
        ha="center",
        va="bottom",
        fontweight="bold",
        color=color,
    )

ax.set_xlabel("Budget Share (%) (Investment)")
ax.set_ylabel("Revenue Share (%) (Return)")
ax.set_title("Genre Drift: Shift in Investment vs Return (Pre to Post COVID)")
ax.grid(True, linestyle=":", alpha=0.6)

ax.text(
    max_val * 0.1,
    max_val * 0.9,
    "High Efficiency\n(Rev > Budget)",
    color=COLORS["increase"],
    alpha=0.5,
    fontweight="bold",
)
ax.text(
    max_val * 0.9,
    max_val * 0.1,
    "Low Efficiency\n(Budget > Rev)",
    color=COLORS["decrease"],
    alpha=0.5,
    fontweight="bold",
    ha="right",
)

plt.xlim(0, max_val)
plt.ylim(0, max_val)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

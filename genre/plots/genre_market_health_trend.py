import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style, COLORS

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/dataset_final.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "genre_market_health_trend.pdf")

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
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    df = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()
else:
    print(f"Error: {INPUT_FILE} not found.")
    exit(1)

genre_rows = []
for idx, row in df.iterrows():
    if not isinstance(row["genres"], list):
        continue

    for g in row["genres"]:
        genre_name = g.get("name") if isinstance(g, dict) else g
        if genre_name == "TV Movie" or not genre_name:
            continue

        genre_rows.append(
            {"year": row["year"], "genre": genre_name, "revenue": row.get("revenue", 0)}
        )

df_flat = pd.DataFrame(genre_rows)

vol_grouped = df_flat.groupby(["year", "genre"]).size().reset_index(name="count")
year_vol_totals = df_flat.groupby("year").size().reset_index(name="total_year_count")
vol_merged = pd.merge(vol_grouped, year_vol_totals, on="year")
vol_merged["volume_share"] = (
    vol_merged["count"] / vol_merged["total_year_count"]
) * 100

rev_grouped = df_flat.groupby(["year", "genre"])["revenue"].sum().reset_index()
year_rev_totals = (
    df_flat.groupby("year")["revenue"]
    .sum()
    .reset_index()
    .rename(columns={"revenue": "total_year_rev"})
)
rev_merged = pd.merge(rev_grouped, year_rev_totals, on="year")
rev_merged["revenue_share"] = (
    rev_merged["revenue"] / rev_merged["total_year_rev"]
) * 100

metrics = pd.merge(
    vol_merged[["year", "genre", "volume_share"]],
    rev_merged[["year", "genre", "revenue_share"]],
    on=["year", "genre"],
    how="outer",
).fillna(0)

total_rev_by_genre = (
    df_flat.groupby("genre")["revenue"].sum().sort_values(ascending=False)
)
top_genres = total_rev_by_genre.head(12).index.tolist()

plot_data = metrics[metrics["genre"].isin(top_genres)].copy()

num_genres = len(top_genres)
cols = 4
rows = (num_genres // cols) + (1 if num_genres % cols > 0 else 0)

set_style(column="full", nrows=rows, ncols=cols)

fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
axes = axes.flatten()

years = sorted(plot_data["year"].unique())

for i, genre in enumerate(top_genres):
    ax = axes[i]
    g_data = plot_data[plot_data["genre"] == genre].sort_values("year")
    g_data = g_data.set_index("year").reindex(years, fill_value=0).reset_index()

    ax.plot(
        g_data["year"],
        g_data["revenue_share"],
        color=COLORS["revenue"],
        linewidth=2.5,
        label="Revenue Share" if i == 0 else "",
    )
    ax.plot(
        g_data["year"],
        g_data["volume_share"],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Volume Share" if i == 0 else "",
    )

    ax.fill_between(
        g_data["year"],
        g_data["revenue_share"],
        g_data["volume_share"],
        where=(g_data["revenue_share"] >= g_data["volume_share"]),
        interpolate=True,
        color=COLORS["increase"],
        alpha=0.2,
    )

    ax.fill_between(
        g_data["year"],
        g_data["revenue_share"],
        g_data["volume_share"],
        where=(g_data["revenue_share"] < g_data["volume_share"]),
        interpolate=True,
        color=COLORS["decrease"],
        alpha=0.2,
    )

    ax.axvline(2019.5, color="black", linestyle=":", alpha=0.5)

    ax.set_title(genre)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if i >= (rows - 1) * cols:
        ax.set_xlabel("Year")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45)

    if i % cols == 0:
        ax.set_ylabel("Share (%)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles = [
    plt.Line2D(
        [], [], color=COLORS["revenue"], linewidth=2.5, label="Revenue Share (Money)"
    ),
    plt.Line2D(
        [],
        [],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Volume Share (Supply)",
    ),
    plt.Rectangle(
        (0, 0), 1, 1, color=COLORS["increase"], alpha=0.2, label="Efficient (Rev > Vol)"
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=COLORS["decrease"],
        alpha=0.2,
        label="Oversupplied (Vol > Rev)",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.02),
    frameon=False,
)

plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.1)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

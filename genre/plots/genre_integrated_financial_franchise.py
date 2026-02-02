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
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "genre_integrated_financial_franchise.pdf"
)

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

    df = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()
else:
    print(f"Error: {INPUT_FILE} not found.")
    sys.exit(1)

genre_rows = []
for idx, row in df.iterrows():
    if not isinstance(row["genres"], list):
        continue

    rev = row.get("revenue", 0)
    budg = row.get("budget", 0)
    is_fran = row.get("is_franchise", 0)

    for g in row["genres"]:
        genre_name = g.get("name") if isinstance(g, dict) else g
        if genre_name == "TV Movie" or not genre_name:
            continue

        genre_rows.append(
            {
                "year": row["year"],
                "genre": genre_name,
                "revenue": rev,
                "franchise_revenue": rev if is_fran else 0,
                "budget": budg,
                "is_franchise": is_fran,
            }
        )

df_flat = pd.DataFrame(genre_rows)

year_totals = (
    df_flat.groupby("year")
    .agg(total_year_rev=("revenue", "sum"), total_year_budg=("budget", "sum"))
    .reset_index()
)

grouped = (
    df_flat.groupby(["year", "genre"])
    .agg(
        genre_total_rev=("revenue", "sum"),
        genre_total_budg=("budget", "sum"),
        genre_franchise_rev=("franchise_revenue", "sum"),
    )
    .reset_index()
)

grouped = pd.merge(grouped, year_totals, on="year")

grouped["total_rev_share"] = (
    grouped["genre_total_rev"] / grouped["total_year_rev"]
) * 100
grouped["total_budg_share"] = (
    grouped["genre_total_budg"] / grouped["total_year_budg"]
) * 100
grouped["franchise_rev_share_global"] = (
    grouped["genre_franchise_rev"] / grouped["total_year_rev"]
) * 100

total_rev_by_genre = (
    df_flat.groupby("genre")["revenue"].sum().sort_values(ascending=False)
)
top_genres = total_rev_by_genre.head(12).index.tolist()

plot_data = grouped[grouped["genre"].isin(top_genres)].copy()

num_genres = len(top_genres)
cols = 4
rows = (num_genres // cols) + (1 if num_genres % cols > 0 else 0)

set_style(column="full", nrows=rows, ncols=cols)

# Ensure all text is the same size as requested
base_size = plt.rcParams["font.size"]
plt.rcParams.update(
    {
        "axes.labelsize": base_size,
        "axes.titlesize": base_size,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
    }
)

COLOR_REV_TOTAL = COLORS["revenue"]
COLOR_REV_FRAN = COLORS["franchise"]
COLOR_BUDG = COLORS["budget"]

fig, axes = plt.subplots(rows, cols, sharex=True)
axes = axes.flatten()

years = sorted(plot_data["year"].unique())

for i, genre in enumerate(top_genres):
    ax = axes[i]
    g_data = plot_data[plot_data["genre"] == genre].sort_values("year")
    g_data = g_data.set_index("year").reindex(years, fill_value=0).reset_index()

    ax.plot(
        g_data["year"],
        g_data["total_rev_share"],
        color=COLOR_REV_TOTAL,
        linewidth=2.5,
        label="Total Rev Share" if i == 0 else "",
        zorder=4,
    )

    ax.plot(
        g_data["year"],
        g_data["franchise_rev_share_global"],
        color=COLOR_REV_FRAN,
        linewidth=2.0,
        linestyle="-",
        label="Franchise Rev (Contribution)" if i == 0 else "",
        zorder=3,
    )

    ax.plot(
        g_data["year"],
        g_data["total_budg_share"],
        color=COLOR_BUDG,
        linewidth=2.0,
        linestyle="--",
        label="Budget Share" if i == 0 else "",
        zorder=5,
    )

    ax.fill_between(
        g_data["year"],
        g_data["franchise_rev_share_global"],
        g_data["total_rev_share"],
        color=COLOR_REV_TOTAL,
        alpha=0.1,
        label="Non-Franchise Rev" if i == 0 else "",
        zorder=1,
    )

    ax.axvline(2019.5, color="black", linestyle=":", alpha=0.3, linewidth=1)

    ax.set_title(genre, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if i >= (rows - 1) * cols:
        ax.set_xlabel("Year")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45)

    if i % cols == 0:
        ax.set_ylabel("Share (%)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles = [
    plt.Line2D([], [], color=COLOR_REV_TOTAL, linewidth=2.5, label="Total Rev Share"),
    plt.Line2D(
        [], [], color=COLOR_BUDG, linewidth=2.0, linestyle="--", label="Budget Share"
    ),
    plt.Line2D(
        [], [], color=COLOR_REV_FRAN, linewidth=2.0, label="Franchise Rev (Contrib)"
    ),
    plt.Rectangle(
        (0, 0), 1, 1, color=COLOR_REV_TOTAL, alpha=0.1, label="Non-Franchise Gap"
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.07),
    frameon=False,
)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

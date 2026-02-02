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
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "genre_franchise_saturation_trend.pdf"
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
                "is_franchise": is_fran,
            }
        )

df_flat = pd.DataFrame(genre_rows)

grouped = (
    df_flat.groupby(["year", "genre"])
    .agg(
        total_count=("is_franchise", "count"),
        franchise_count=("is_franchise", "sum"),
        total_revenue=("revenue", "sum"),
        franchise_revenue=(
            "revenue",
            lambda x: x[df_flat.loc[x.index, "is_franchise"] == 1].sum(),
        ),
    )
    .reset_index()
)

grouped["franchise_vol_share"] = (
    grouped["franchise_count"] / grouped["total_count"]
) * 100
grouped["franchise_rev_share"] = 0.0
mask = grouped["total_revenue"] > 0
grouped.loc[mask, "franchise_rev_share"] = (
    grouped.loc[mask, "franchise_revenue"] / grouped.loc[mask, "total_revenue"]
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

fig, axes = plt.subplots(rows, cols, sharex=True)
axes = axes.flatten()

years = sorted(plot_data["year"].unique())

for i, genre in enumerate(top_genres):
    ax = axes[i]
    g_data = plot_data[plot_data["genre"] == genre].sort_values("year")
    g_data = g_data.set_index("year").reindex(years, fill_value=0).reset_index()

    ax.plot(
        g_data["year"],
        g_data["franchise_rev_share"],
        color=COLORS["franchise"],
        linewidth=2.5,
        label="Franchise Rev %" if i == 0 else "",
    )
    ax.plot(
        g_data["year"],
        g_data["franchise_vol_share"],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Franchise Vol %" if i == 0 else "",
    )

    ax.fill_between(
        g_data["year"],
        g_data["franchise_rev_share"],
        g_data["franchise_vol_share"],
        where=(g_data["franchise_rev_share"] >= g_data["franchise_vol_share"]),
        interpolate=True,
        color=COLORS["increase"],
        alpha=0.2,
    )

    ax.fill_between(
        g_data["year"],
        g_data["franchise_rev_share"],
        g_data["franchise_vol_share"],
        where=(g_data["franchise_rev_share"] < g_data["franchise_vol_share"]),
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
        ax.set_ylabel("Franchise %")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles = [
    plt.Line2D(
        [], [], color=COLORS["franchise"], linewidth=2.5, label="Franchise Revenue %"
    ),
    plt.Line2D(
        [],
        [],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Franchise Volume %",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=COLORS["increase"],
        alpha=0.2,
        label="High Yield (Rev > Vol)",
    ),
    plt.Rectangle(
        (0, 0), 1, 1, color=COLORS["decrease"], alpha=0.2, label="Low Yield (Vol > Rev)"
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.05),
    frameon=False,
)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

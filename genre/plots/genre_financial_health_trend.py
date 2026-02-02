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
    os.path.dirname(__file__), "genre_financial_health_trend.pdf"
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
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    df = df[(df["vote_count"] >= 10) & (df["revenue"] > 0) & (df["budget"] > 0)].copy()
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
            {
                "year": row["year"],
                "genre": genre_name,
                "revenue": row.get("revenue", 0),
                "budget": row.get("budget", 0),
            }
        )

df_flat = pd.DataFrame(genre_rows)
df_flat = df_flat[(df_flat["year"] >= 2000) & (df_flat["year"] <= 2024)]

year_grouped = (
    df_flat.groupby(["year", "genre"])
    .agg(total_rev=("revenue", "sum"), total_budg=("budget", "sum"))
    .reset_index()
)

year_totals = (
    df_flat.groupby("year")
    .agg(ind_rev=("revenue", "sum"), ind_budg=("budget", "sum"))
    .reset_index()
)

merged = pd.merge(year_grouped, year_totals, on="year")
merged["revenue_share"] = (merged["total_rev"] / merged["ind_rev"]) * 100
merged["budget_share"] = (merged["total_budg"] / merged["ind_budg"]) * 100

merged["year_dt"] = pd.to_datetime(merged["year"], format="%Y")

top_genres = (
    df_flat.groupby("genre")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .index.tolist()
)

cols = 4
rows = 3
set_style(column="full", nrows=rows, ncols=cols)

fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
axes = axes.flatten()

for i, genre in enumerate(top_genres):
    ax = axes[i]
    g_data = merged[merged["genre"] == genre].sort_values("year")

    ax.plot(
        g_data["year"],
        g_data["revenue_share"],
        color=COLORS["revenue"],
        linewidth=2.5,
        label="Revenue Share" if i == 0 else "",
    )
    ax.plot(
        g_data["year"],
        g_data["budget_share"],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Budget Share" if i == 0 else "",
    )

    ax.fill_between(
        g_data["year"],
        g_data["revenue_share"],
        g_data["budget_share"],
        where=(g_data["revenue_share"] >= g_data["budget_share"]),
        interpolate=True,
        color=COLORS["increase"],
        alpha=0.2,
    )

    ax.fill_between(
        g_data["year"],
        g_data["revenue_share"],
        g_data["budget_share"],
        where=(g_data["revenue_share"] < g_data["budget_share"]),
        interpolate=True,
        color=COLORS["decrease"],
        alpha=0.2,
    )

    ax.set_title(genre)
    ax.grid(True, linestyle=":", alpha=0.3)

    if i % cols == 0:
        ax.set_ylabel("Share (%)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles = [
    plt.Line2D(
        [], [], color=COLORS["revenue"], linewidth=2.5, label="Revenue Share (Return)"
    ),
    plt.Line2D(
        [],
        [],
        color=COLORS["budget"],
        linewidth=2.5,
        linestyle="--",
        label="Budget Share (Investment)",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=COLORS["increase"],
        alpha=0.2,
        label="High ROI (Rev > Budget)",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=COLORS["decrease"],
        alpha=0.2,
        label="Low ROI (Budget > Rev)",
    ),
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=4,
    frameon=False,
)

fig.text(
    0.5,
    0.08,
    "Figure 2: Financial Health Trend—Comparing Revenue Share vs Budget Share.\nGreen areas indicate efficient capital use (Rev > Budget); Red areas indicate over-investment (Budget > Rev).",
    ha="center",
    style="italic",
)

plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.1)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

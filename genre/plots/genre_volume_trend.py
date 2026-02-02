import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/new_tmdb_movies_master.jsonl"
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "genre_volume_trend.pdf")


print("Loading data...")
data_list = []
if os.path.exists(INPUT_FILE):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data_list.append(json.loads(line))
                except:
                    continue
else:
    print(f"Error: {INPUT_FILE} not found.")
    exit(1)

df = pd.DataFrame(data_list)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = df["release_date"].dt.year


df = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()


genre_rows = []
for idx, row in df.iterrows():
    if not isinstance(row["genres"], list):
        continue

    for g in row["genres"]:
        genre_name = g["name"]
        if genre_name == "TV Movie":
            continue

        genre_rows.append({"year": row["year"], "genre": genre_name})

df_flat = pd.DataFrame(genre_rows)


grouped = df_flat.groupby(["year", "genre"]).size().reset_index(name="count")


year_totals = df_flat.groupby("year").size().reset_index(name="total_year_count")


merged = pd.merge(grouped, year_totals, on="year")
merged["volume_share"] = (merged["count"] / merged["total_year_count"]) * 100


total_vol_by_genre = df_flat.groupby("genre").size().sort_values(ascending=False)
top_genres = total_vol_by_genre.head(12).index.tolist()

plot_data = merged[merged["genre"].isin(top_genres)].copy()


print("Plotting...")

ncols = 3
nrows = (len(top_genres) // ncols) + (1 if len(top_genres) % ncols > 0 else 0)
set_style(column="full", nrows=nrows, ncols=ncols)

g = sns.FacetGrid(
    plot_data, col="genre", col_wrap=3, height=3, aspect=1.5, sharey=False
)

g.map(sns.lineplot, "year", "volume_share", linewidth=2.5, marker="o", color="#d62728")


for ax in g.axes.flat:
    ax.axvline(2019.5, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.tick_params(axis="x", rotation=45)

g.set_titles("{col_name}")
g.set_axis_labels("Year", "Volume Share (%)")


plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

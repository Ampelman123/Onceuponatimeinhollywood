import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style

set_style()

INPUT_FILE = os.path.join(os.path.dirname(__file__), "../../data/raw_dataset_arda.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "genre_correlation_matrix.pdf")
N_BOOTSTRAP = 1000

if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit(1)

df = pd.read_csv(INPUT_FILE)


def safe_parse_genres(val):
    if isinstance(val, str) and val.strip():
        try:
            return ast.literal_eval(val)
        except:
            return []
    return []


df["genres"] = df["genres"].apply(safe_parse_genres)

df_exploded = df.explode("genres")
df_exploded["genre_name"] = df_exploded["genres"].apply(
    lambda x: x["name"] if isinstance(x, dict) else None
)
df_exploded = df_exploded.dropna(subset=["genre_name"])
df_exploded = df_exploded[df_exploded["genre_name"] != "TV Movie"]

df_exploded["movie_index"] = df_exploded.index
genre_matrix = pd.crosstab(df_exploded.index, df_exploded["genre_name"])

valid_genres = genre_matrix.sum().sort_values(ascending=False).index
genre_matrix = genre_matrix[valid_genres]

obs_corr = genre_matrix.corr(method="pearson")

bootstrap_matrices = []
n_samples = len(genre_matrix)

for i in range(N_BOOTSTRAP):
    if i % 100 == 0:
        print(f"Iteration {i}/{N_BOOTSTRAP}...")

    sample_indices = np.random.choice(genre_matrix.index, size=n_samples, replace=True)
    sample_df = genre_matrix.loc[sample_indices]
    corr = sample_df.corr(method="pearson")
    bootstrap_matrices.append(corr.values)

bootstrap_array = np.array(bootstrap_matrices)
mean_corr_values = np.mean(bootstrap_array, axis=0)

mean_corr_matrix = pd.DataFrame(
    mean_corr_values, index=obs_corr.index, columns=obs_corr.columns
)

fig, ax = plt.subplots()

mask = np.triu(np.ones_like(mean_corr_matrix, dtype=bool))

sns.heatmap(
    mean_corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "shrink": 0.7,
        "label": f"Mean Pearson Correlation ({N_BOOTSTRAP} Bootstraps)",
    },
    ax=ax,
)

plt.title(f"Genre Co-occurrence: Mean Bootstrap Correlation (n={N_BOOTSTRAP})")
plt.xlabel("")
plt.ylabel("")

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")

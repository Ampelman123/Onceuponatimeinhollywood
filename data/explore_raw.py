# This file is mostly obsolete, as we now have a true streaming column via API, but was used to dump Jan 1. deletion and basic data exploration

import pandas as pd
import json
import numpy as np

# --- CONFIGURATION ---
FILE_PATH = "new_tmdb_movies_master.jsonl"  # Use your actual filename
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Streaming Keywords
STREAMING_KEYWORDS = [
    "Netflix",
    "Amazon Studios",
    "Hulu",
    "Apple TV+",
    "Disney+",
    "HBO Max",
    "Prime Video",
    "Paramount+",
    "Warner Bros. Television",
    "Sky",
]

print("Loading raw data line-by-line...")

data_list = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data_list.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Create DataFrame
df_raw = pd.DataFrame(data_list)
print(f"Loaded {len(df_raw)} raw rows.")

# ---------------------------------------------------------
# 1. DATE PARSING & FILTERING
# ---------------------------------------------------------
df_raw["release_date"] = pd.to_datetime(df_raw["release_date"], errors="coerce")

# Filter for your Time Window
df_raw = df_raw[
    (df_raw["release_date"] >= START_DATE) & (df_raw["release_date"] <= END_DATE)
].copy()

# --- DETECT THE "JANUARY 1st" ARTIFACT ---
# Create a flag for movies released on Jan 1st
df_raw["is_jan_1"] = (df_raw["release_date"].dt.month == 1) & (
    df_raw["release_date"].dt.day == 1
)

print(
    f"\nPotential 'Dump Date' Artifacts (Jan 1st releases): {df_raw['is_jan_1'].sum()}"
)
print(
    "If this number is huge (e.g., >500), these are likely database defaults, not real releases."
)

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------
# Convert numerics
cols_to_numeric = ["budget", "revenue", "vote_count", "vote_average", "popularity"]
for col in cols_to_numeric:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

# Calculate ROI (Movie Level)
# Only calculate for movies with valid budgets to avoid Inf/NaN
valid_budget = df_raw["budget"] > 1000  # Threshold to avoid micro-budget noise
df_raw["roi"] = np.where(
    valid_budget, (df_raw["revenue"] - df_raw["budget"]) / df_raw["budget"], np.nan
)


# Extract First Production Company (for quick checks)
def get_main_company(companies):
    if isinstance(companies, list) and len(companies) > 0:
        return companies[0].get("name", "Unknown")
    return "Unknown"


df_raw["main_company"] = df_raw["production_companies"].apply(get_main_company)


# Flag Streaming
def is_streaming_prod(companies):
    if not isinstance(companies, list):
        return False
    txt = " ".join([c.get("name", "").lower() for c in companies])
    return any(k.lower() in txt for k in STREAMING_KEYWORDS)


df_raw["is_streaming"] = df_raw["production_companies"].apply(is_streaming_prod)

# ---------------------------------------------------------
# 3. PREVIEW & INSPECTION
# ---------------------------------------------------------
# Show the "Blockbuster vs Arthouse" examples (High Rev, Low Rating)
print("\n--- Top 5 High Revenue but Low Rating (< 6.0) ---")
high_rev_low_rate = (
    df_raw[(df_raw["revenue"] > 100_000_000) & (df_raw["vote_average"] < 6.0)]
    .sort_values("revenue", ascending=False)
    .head(5)
)
print(
    high_rev_low_rate[
        ["title", "release_date", "revenue", "vote_average", "main_company"]
    ]
)

print("\n--- Top 5 Streaming Movies by Popularity ---")
streaming_hits = (
    df_raw[df_raw["is_streaming"]].sort_values("popularity", ascending=False).head(5)
)
print(streaming_hits[["title", "release_date", "vote_average", "main_company"]])

print("\n--- Dataframe ready as 'df_raw' ---")

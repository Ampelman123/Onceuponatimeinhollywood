import pandas as pd
import numpy as np
import json
import ast

# --- CONFIGURATION ---
# Pointing to the cleaned CSV file created in the previous step
FILE_PATH = "/Users/hagencarstensen/Documents/WiSe25/Data Literacy/Project/Code/DataLiteracyProject/data/movies_no_jan_1.csv"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Intervention Dates (Must be pd.Timestamp objects for comparison)
SHOCK_DATE = pd.to_datetime("2020-03-01")  # Start of Covid Shock
ADJUSTMENT_DATE = pd.to_datetime("2022-01-01")  # Start of Adjustment

# Thresholds to remove noise
MIN_VOTE_COUNT = 50  # Min votes for reliable rating
MIN_BUDGET = 10000  # Min budget for financial analysis
MIN_REVENUE = 10000  # Min revenue for financial analysis

# Streaming Keywords (Production Companies)
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

# ---------------------------------------------------------
# STEP 1: LOAD DATA FROM CSV
# ---------------------------------------------------------
print("Loading data from CSV...")

df = pd.read_csv(FILE_PATH)

# Parse 'production_companies' from string representation to list of dicts
# This is necessary because CSVs store lists/dicts as strings
if "production_companies" in df.columns:
    df["production_companies"] = df["production_companies"].apply(
        lambda x: (
            ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
    )

print(f"Successfully loaded {len(df)} movies.")

# Clean up core columns and filter date range
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"]).copy()
df = df[(df["release_date"] >= START_DATE) & (df["release_date"] <= END_DATE)].copy()

# Ensure numeric columns are handled
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (MOVIE LEVEL)
# ---------------------------------------------------------
print("Engineering features...")

df["valid_finance"] = (df["budget"] >= MIN_BUDGET) & (df["revenue"] >= MIN_REVENUE)
df["valid_stats"] = df["vote_count"] >= MIN_VOTE_COUNT

df["roi"] = np.where(
    df["valid_finance"], (df["revenue"] - df["budget"]) / df["budget"], np.nan
)
df["avg_log_roi"] = np.where(df["valid_finance"], np.log(df["roi"] + 1 + 1e-9), np.nan)


def is_streaming(companies_list):
    if not isinstance(companies_list, list) or not companies_list:
        return 0
    # Ensure elements are dicts before accessing .get
    company_names = " ".join(
        [c.get("name", "").lower() for c in companies_list if isinstance(c, dict)]
    )
    return (
        1
        if any(keyword.lower() in company_names for keyword in STREAMING_KEYWORDS)
        else 0
    )


df["is_streaming"] = df["production_companies"].apply(is_streaming).astype(int)
df["vote_average_filtered"] = np.where(df["valid_stats"], df["vote_average"], np.nan)

# ---------------------------------------------------------
# STEP 3: AGGREGATE TO MONTHLY TIME SERIES
# ---------------------------------------------------------
print("Aggregating to monthly time series...")

df.set_index("release_date", inplace=True)

agg_rules = {
    "title": "count",
    "vote_average_filtered": "mean",
    "avg_log_roi": "mean",
    "is_streaming": "mean",
}

ts_df = df.resample("MS").agg(agg_rules)
ts_df.columns = ["movie_count", "avg_rating", "avg_log_roi", "prop_streaming"]
ts_df["movie_count"] = ts_df["movie_count"].fillna(0)

# ---------------------------------------------------------
# STEP 4: CREATE ITS VARIABLES
# ---------------------------------------------------------
print("Creating ITS regression variables...")

ts_df["time_index"] = np.arange(len(ts_df))

shock_start_idx = ts_df[ts_df.index >= SHOCK_DATE]["time_index"].min()
adjust_start_idx = ts_df[ts_df.index >= ADJUSTMENT_DATE]["time_index"].min()

ts_df["D1_shock"] = (ts_df.index >= SHOCK_DATE).astype(int)
ts_df["P1_shock_slope"] = np.where(
    ts_df["D1_shock"] == 1, ts_df["time_index"] - shock_start_idx, 0
)
ts_df["D2_adjust"] = (ts_df.index >= ADJUSTMENT_DATE).astype(int)
ts_df["P2_adjust_slope"] = np.where(
    ts_df["D2_adjust"] == 1, ts_df["time_index"] - adjust_start_idx, 0
)
ts_df["month"] = ts_df.index.month

ts_df.to_csv("movie_time_series_cleaned.csv")
print("Saved aggregated time series to movie_time_series_cleaned.csv")

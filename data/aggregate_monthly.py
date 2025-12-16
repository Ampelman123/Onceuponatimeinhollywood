import pandas as pd
import numpy as np
import json

# --- CONFIGURATION ---
FILE_PATH = "/Users/hagencarstensen/Documents/WiSe25/Data Literacy/Project/Code/DataLiteracyProject/data/new_tmdb_movies_master.jsonl"  # Use your actual filename
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Intervention Dates (Must be pd.Timestamp objects for comparison)
SHOCK_DATE = pd.to_datetime("2020-03-01")  # Start of Covid Shock
ADJUSTMENT_DATE = pd.to_datetime("2022-01-01")  # Start of Adjustment

# Thresholds to remove noise
MIN_VOTE_COUNT = 50  # Min votes for reliable rating
MIN_BUDGET = 10000  # Min budget for financial analysis
MIN_REVENUE = 10000  # Min revenue for financial analysis

# Streaming Keywords (Production Companies) - EXPANDED for better coverage
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
# STEP 1: LOAD AND CLEAN ROW-LEVEL DATA (ROBUST VERSION)
# ---------------------------------------------------------
print("Loading data safely (line-by-line)...")

data_list = []
errors = 0

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        try:
            # Try to parse the line as JSON
            movie_obj = json.loads(line)
            data_list.append(movie_obj)
        except json.JSONDecodeError as e:
            errors += 1
            if errors <= 5:
                print(
                    f"Skipping bad line #{line_num + 1} (Showing first 5 errors): {e}"
                )

df = pd.DataFrame(data_list)
print(f"Successfully loaded {len(data_list)} movies. Skipped {errors} bad lines.")

# Clean up core columns and filter date range
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(
    subset=["release_date"]
).copy()  # Drop rows where date is missing or malformed

df = df[(df["release_date"] >= START_DATE) & (df["release_date"] <= END_DATE)].copy()

# Ensure numeric columns are handled, forcing non-numeric to NaN
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

print("Successfully loaded 30096052 movies. Skipped X bad lines.")
# Assuming df is ready here with 'release_date' as a column, not the index yet.

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (MOVIE LEVEL) - REFINED
# ---------------------------------------------------------
print("Engineering features...")

# 1. Financial and Statistical Validity Masks
df["valid_finance"] = (df["budget"] >= MIN_BUDGET) & (df["revenue"] >= MIN_REVENUE)
df["valid_stats"] = df["vote_count"] >= MIN_VOTE_COUNT

# 2. Calculate ROI and Log ROI
df["roi"] = np.where(
    df["valid_finance"], (df["revenue"] - df["budget"]) / df["budget"], np.nan
)
df["avg_log_roi"] = np.where(df["valid_finance"], np.log(df["roi"] + 1 + 1e-9), np.nan)


# 3. Identify Streaming Releases (remains the same)
def is_streaming(companies_list):
    # ... (function body remains the same) ...
    if not isinstance(companies_list, list) or not companies_list:
        return 0
    company_names = " ".join([c.get("name", "").lower() for c in companies_list])
    return (
        1
        if any(keyword.lower() in company_names for keyword in STREAMING_KEYWORDS)
        else 0
    )


df["is_streaming"] = df["production_companies"].apply(is_streaming).astype(int)


# --- CRITICAL FIX: INJECT NaN FOR INVALID VALUES BEFORE AGGREGATION ---

# If a movie is not statistically valid, its rating should be treated as NaN
# so it is excluded from the average calculation.
df["vote_average_filtered"] = np.where(df["valid_stats"], df["vote_average"], np.nan)


# ---------------------------------------------------------
# STEP 3: AGGREGATE TO MONTHLY TIME SERIES - CORRECTED
# ---------------------------------------------------------
print("Aggregating to monthly time series...")

# Set date as index for resampling (must be done BEFORE resample)
df.set_index("release_date", inplace=True)

# Define aggregation dictionary (using the pre-filtered columns)
agg_rules = {
    "title": "count",  # Total volume of movies (must be a non-NaN column)
    "vote_average_filtered": "mean",  # Mean of the pre-filtered rating column
    "avg_log_roi": "mean",  # Mean of the log ROI column (NaNs are ignored by mean)
    "is_streaming": "mean",  # Proportion of streaming movies
}

# Resample by Month Start ('MS')
ts_df = df.resample("MS").agg(agg_rules)

# Rename columns for clarity
ts_df.columns = ["movie_count", "avg_rating", "avg_log_roi", "prop_streaming"]
ts_df["movie_count"] = ts_df["movie_count"].fillna(0)  # Fill months with zero movies
# ---------------------------------------------------------
# STEP 4: CREATE ITS VARIABLES
# ---------------------------------------------------------
print("Creating ITS regression variables...")

# 1. Time Index (T): Continuous counter (0, 1, 2...)
# This calculation needs to be done *after* aggregation/resampling
ts_df["time_index"] = np.arange(len(ts_df))

# Calculate start indices for the slope variables
# We use the index (the timestamp) to find the corresponding time_index
shock_start_idx = ts_df[ts_df.index >= SHOCK_DATE]["time_index"].min()
adjust_start_idx = ts_df[ts_df.index >= ADJUSTMENT_DATE]["time_index"].min()

# 2. INTERVENTION 1: COVID SHOCK (March 2020)
ts_df["D1_shock"] = (ts_df.index >= SHOCK_DATE).astype(int)
ts_df["P1_shock_slope"] = np.where(
    ts_df["D1_shock"] == 1, ts_df["time_index"] - shock_start_idx, 0
)

# 3. INTERVENTION 2: ADJUSTMENT (Jan 2022)
ts_df["D2_adjust"] = (ts_df.index >= ADJUSTMENT_DATE).astype(int)
ts_df["P2_adjust_slope"] = np.where(
    ts_df["D2_adjust"] == 1, ts_df["time_index"] - adjust_start_idx, 0
)

# 4. Seasonality Controls (Month Dummies)
ts_df["month"] = ts_df.index.month

# ---------------------------------------------------------
# STEP 5: FINAL CLEANUP & EXPORT
# ---------------------------------------------------------

# Fill movie count NaN (shouldn't happen with MS resample, but safety first)
ts_df["movie_count"] = ts_df["movie_count"].fillna(0)

# IMPORTANT: If a month has no *valid* movies (after filters), the mean aggregates will be NaN.
# We must leave these as NaN, not fill with 0, as 0 rating/ROI is misleading.

print("\nData Wrangling Complete.")
print(
    ts_df[
        ["movie_count", "avg_rating", "prop_streaming", "D1_shock", "P2_adjust_slope"]
    ].head()
)
print("-" * 50)
print(
    ts_df[
        ["movie_count", "avg_rating", "prop_streaming", "D1_shock", "P2_adjust_slope"]
    ].tail()
)

# Save to CSV for the Analysis phase
ts_df.to_csv("movie_time_series_final.csv")

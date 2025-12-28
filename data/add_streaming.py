# Poll API for Release Details

import pandas as pd
import requests
import time
import numpy as np
from tqdm import tqdm  # Progress bar library

# --- CONFIGURATION ---
API_KEY = "50fabac2bdba5ae592a69cf47567f3f4"  # Paste your key here
INPUT_FILE = "new_tmdb_movies_master.jsonl"
OUTPUT_FILE = "movies_with_release_type.csv"

# Load your raw data (we'll re-apply the basic filters to save time)
print("Loading and filtering raw data...")
data_list = []
import json

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                data_list.append(json.loads(line))
            except:
                continue

df = pd.DataFrame(data_list)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")


print(f"Enriching {len(df)} movies. This may take time...")


# --- RELEASE DATE LOGIC ---
def get_release_class(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates"
    try:
        r = requests.get(url, params={"api_key": API_KEY}, timeout=5)
        if r.status_code != 200:
            return "unknown"

        data = r.json()

        # 1. Focus on US release (Proxy for primary market strategy)
        # If US is missing, fallback to the first country listed
        us_data = next(
            (x for x in data.get("results", []) if x["iso_3166_1"] == "US"), None
        )
        if not us_data and data.get("results"):
            us_data = data["results"][0]  # Fallback

        if not us_data:
            return "unknown"

        dates = us_data["release_dates"]

        # 2. Extract earliest dates for Theatrical (3) and Digital (4)
        # We use a default of "None" if the type doesn't exist
        theatrical_dates = [d["release_date"] for d in dates if d["type"] == 3]
        digital_dates = [d["release_date"] for d in dates if d["type"] == 4]

        t_date = min(theatrical_dates) if theatrical_dates else None
        d_date = min(digital_dates) if digital_dates else None

        # 3. CLASSIFICATION LOGIC

        # CASE A: Only Digital exists -> Streaming
        if d_date and not t_date:
            return "streaming"

        # CASE B: Both exist -> Compare dates
        if d_date and t_date:
            # If Digital is BEFORE or SAME DAY as Theatrical -> Streaming (Day & Date)
            if d_date <= t_date:
                return "streaming"
            # If Digital is within 14 days of Theatrical -> Hybrid/Streaming-first
            # (You can tune this buffer window)
            elif (pd.to_datetime(d_date) - pd.to_datetime(t_date)).days < 14:
                return "streaming"
            # Otherwise -> Theatrical
            else:
                return "theatrical"

        # CASE C: Only Theatrical exists -> Theatrical
        if t_date:
            return "theatrical"

        return "unknown"

    except Exception as e:
        return "error"


# --- RUN LOOP WITH PROGRESS BAR ---
tqdm.pandas()  # Initialize progress bar for pandas
df["release_strategy"] = df["id"].progress_apply(get_release_class)

# Convert to boolean for your regression (1 = Streaming, 0 = Theatrical)
# We treat 'unknown'/'error' as 0 (conservative approach)
df["is_streaming_verified"] = (df["release_strategy"] == "streaming").astype(int)

# Drop movies released on January 1st
initial_rows = len(df)
df = df[~((df["release_date"].dt.month == 1) & (df["release_date"].dt.day == 1))]
rows_after_drop = len(df)
print(f"Dropped {initial_rows - rows_after_drop} movies released on January 1st.")

# --- SAVE ---
print(f"Finished. Detected {df['is_streaming_verified'].sum()} streaming releases.")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved enriched data to {OUTPUT_FILE}")

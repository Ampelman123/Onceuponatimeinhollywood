# Add gini coeficient for revenue and shannon entropy for genre diversity
import pandas as pd
import numpy as np
import json

# Load your enriched movie data
df = pd.read_csv("movies_with_release_type.csv")
df["release_date"] = pd.to_datetime(df["release_date"])

# --- HELPER FUNCTIONS ---


def calculate_gini(incomes):
    """Calculates Gini Coefficient for a list of revenues (inequality)."""
    incomes = np.sort(np.array(incomes))
    n = len(incomes)
    if n == 0 or np.sum(incomes) == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * incomes).sum() / (n * np.sum(incomes))


def calculate_entropy(genre_lists):
    """Calculates Shannon Entropy for genre diversity."""
    # Flatten the list of lists: [['Action', 'Comedy'], ['Action']] -> ['Action', 'Comedy', 'Action']
    all_genres = []
    for g_list in genre_lists:
        if isinstance(g_list, str):  # Handle string representation of lists
            try:
                # Parse stringified list "['Action', 'Comedy']" -> list
                # (You might need json.loads or ast.literal_eval depending on format)
                # For safety, let's assume standard extraction has happened.
                # Simplest fallback if it's just a raw list of dicts string:
                pass
            except:
                pass

        # Assuming you extracted genres into a simple list column earlier
        # If not, add extraction logic here.
        if isinstance(g_list, list):
            all_genres.extend(g_list)

    if not all_genres:
        return np.nan

    # Calculate probabilities
    counts = pd.Series(all_genres).value_counts()
    probs = counts / counts.sum()

    # Shannon Entropy formula: -Sum(p * log(p))
    return -np.sum(probs * np.log2(probs))


# --- DATA PREP (Extract Genres & Collection) ---


# 1. Parse 'genres' from JSON string if needed
# (Assuming your wrangling script kept them as lists or you re-parse them)
# For this example, let's assume you have a column 'genre_names' which is a list of strings
# You might need to adjust this block to match your specific dataframe structure
def extract_genre_names(row):
    try:
        # If it's a string, eval it or json load it
        if isinstance(row, str):
            row = json.loads(row.replace("'", '"'))  # Basic cleanup
        return [g["name"] for g in row]
    except:
        return []


if "genres" in df.columns and isinstance(df["genres"].iloc[0], str):
    df["genre_list"] = df["genres"].apply(extract_genre_names)

# 2. Flag Sequels
# 'belongs_to_collection' is usually a JSON object or Null
df["is_sequel"] = df["belongs_to_collection"].notnull().astype(int)

# --- MONTHLY AGGREGATION ---
df.set_index("release_date", inplace=True)

creative_ts = df.resample("MS").agg(
    {
        # 1. Inequality Metric (Gini of Revenue)
        "revenue": lambda x: calculate_gini(x),
        # 2. Diversity Metric (Entropy of Genres)
        "genre_list": lambda x: calculate_entropy(x),
        # 3. Structural Metric (Sequel Dominance)
        "is_sequel": "mean",  # % of movies that are sequels
        # Standard Metrics
        "title": "count",
    }
)

creative_ts.columns = ["revenue_gini", "genre_entropy", "prop_sequels", "movie_count"]

# Filter noise
creative_ts = creative_ts[
    creative_ts["movie_count"] > 5
]  # Ignore months with almost no data

print(creative_ts.head())
creative_ts.to_csv("advanced_feature_time_series.csv")

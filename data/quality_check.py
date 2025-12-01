"""
    Performs an Exploratory Data Analysis (EDA) on the raw TMDB dataset.
    It assesses data quality by identifying:
    1. Missing values (NaN)
    2. Integrity issues (e.g., entries with 0 budget/revenue)
    3. Plausibility issues (outliers/micro-budgets)
    4. Temporal consistency (distribution over years)

    Goal: Determine the valid sample size for the final ROI analysis.
    
    Github Issue #1
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import os

INPUT_FILE = "tmdb_movies_master.jsonl"
MIN_BUDGET_THRESHOLD = 10000  # Filter low-budget outliers

def load_jsonl(filename):
    """Parses JSONL file into a list."""
    data = []
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return pd.DataFrame()
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue 
    return pd.DataFrame(data)

def analyze_data():
    print("--- START GITHUB ISSUE #1: DATA QUALITY CHECK ---")
    
    print(f"Loading {INPUT_FILE}...")
    df = load_jsonl(INPUT_FILE)
    
    if df.empty:
        return

    print(f"Total records loaded: {len(df)}")
    
    # Select relevant columns
    cols = ['id', 'title', 'release_date', 'budget', 'revenue', 'vote_average', 'vote_count']
    df = df[[c for c in cols if c in df.columns]]

    # Data Type Conversion
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)

    # Check: Missing Values
    print("\n[1] MISSING VALUES (NaN):")
    print(df.isnull().sum())

    # Check: Zero Values
    zero_budget = df[df['budget'] == 0].shape[0]
    zero_revenue = df[df['revenue'] == 0].shape[0]
    
    print("\n[2] FINANCIAL DATA INTEGRITY:")
    print(f"Budget = 0:  {zero_budget} ({round(zero_budget/len(df)*100, 1)}%)")
    print(f"Revenue = 0: {zero_revenue} ({round(zero_revenue/len(df)*100, 1)}%)")

    # Check: Outliers (Plausibility)
    #  < $10k
    suspicious_budget = df[(df['budget'] > 0) & (df['budget'] < MIN_BUDGET_THRESHOLD)]
    
    print(f"\n[3] OUTLIERS (< ${MIN_BUDGET_THRESHOLD}):")
    print(f"Count: {len(suspicious_budget)}")
    
    # Check: Temporal Distribution
    df['year'] = df['release_date'].dt.year
    movies_per_year = df['year'].value_counts().sort_index()
    
    print("\n[4] YEARLY DISTRIBUTION (Last 10 Years):")
    print(movies_per_year.tail(10))

    clean_financials = df[
        (df['budget'] >= MIN_BUDGET_THRESHOLD) & 
        (df['revenue'] > 0)
    ]
    
    print("\n" + "="*40)
    print("DATA CLEANING SUMMARY")
    print("="*40)
    print(f"Total raw records:   {len(df)}")
    print(f"Valid for ROI stats: {len(clean_financials)}")
    print(f"Data loss:           {len(df) - len(clean_financials)} records")
    
    # Optional Visualization
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(clean_financials['budget'], bins=50, color='blue', alpha=0.7)
        plt.title('Budget Distribution (Valid Data Only)')
        plt.xlabel('Budget (USD)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    except:
        pass

if __name__ == "__main__":
    analyze_data()
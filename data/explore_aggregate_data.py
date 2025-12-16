import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# --- CONFIGURATION ---
INPUT_FILE = "movie_time_series_cleaned.csv"

# Intervention Dates (Must match your wrangling script)
SHOCK_DATE = pd.to_datetime("2020-03-01")
ADJUSTMENT_DATE = pd.to_datetime("2022-01-01")

# Set plot style
sns.set_theme(
    style="whitegrid", context="talk"
)  # 'talk' context makes fonts bigger for papers
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 2.5

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print(f"Loading {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)

    # Check if 'release_date' is a column or the index.
    # If it was saved with to_csv(), it's usually the first column.
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"])
        df.set_index("release_date", inplace=True)
    else:
        # If the index didn't get a name, try the first column
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    print("Data loaded successfully.")
    print(df.head())

except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Make sure it's in the same folder.")
    exit()


# ---------------------------------------------------------
# 2. PLOTTING HELPER FUNCTION
# ---------------------------------------------------------
def plot_its_metric(data, column, title, ylabel, color, y_limit=None):
    """
    Generates a standardized time-series plot with intervention lines.
    """
    fig, ax = plt.subplots()

    # 1. Plot the raw monthly data (lighter line)
    ax.plot(
        data.index,
        data[column],
        color=color,
        alpha=0.4,
        linewidth=1.5,
        label="Monthly Data",
    )

    # 2. Plot a 6-month Rolling Average (solid, darker line) to show the trend
    rolling_mean = data[column].rolling(window=6, center=True).mean()
    ax.plot(data.index, rolling_mean, color=color, linewidth=3, label="6-Month Trend")

    # 3. Add Vertical Intervention Lines
    ax.axvline(
        x=SHOCK_DATE, color="red", linestyle="--", linewidth=2, label="COVID Shock"
    )
    ax.axvline(
        x=ADJUSTMENT_DATE,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Adjustment",
    )

    # 4. Add Shaded Backgrounds for Regimes
    # Pre-COVID
    ax.axvspan(data.index.min(), SHOCK_DATE, alpha=0.05, color="gray")
    # Shock Period
    ax.axvspan(SHOCK_DATE, ADJUSTMENT_DATE, alpha=0.1, color="red")
    # Adjustment Period
    ax.axvspan(ADJUSTMENT_DATE, data.index.max(), alpha=0.1, color="green")

    # Formatting
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("")

    if y_limit:
        ax.set_ylim(y_limit)

    # Date Formatting on X-Axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Save the plot
    filename = f"plot_{column}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.show()


# ---------------------------------------------------------
# 3. GENERATE THE MAIN PLOTS
# ---------------------------------------------------------

# Plot 1: Volume (Did the industry shrink?)
plot_its_metric(
    df,
    "movie_count",
    "Figure 1: Monthly Movie Releases (Volume)",
    "Number of Movies",
    "#1f77b4",  # Matplotlib Blue
)

# Plot 2: Ratings (Did users get happier/grumpier?)
plot_its_metric(
    df,
    "avg_rating",
    "Figure 2: Average User Rating (filtered for quality)",
    "Vote Average (0-10)",
    "#ff7f0e",  # Matplotlib Orange
)

# Plot 3: ROI (Did movies lose money?)
plot_its_metric(
    df,
    "avg_log_roi",
    "Figure 3: Financial Health (Log ROI)",
    "Log ROI (0 = Break Even)",
    "#2ca02c",  # Matplotlib Green
)

# Plot 4: Streaming (Did Netflix take over?)
plot_its_metric(
    df,
    "prop_streaming",
    "Figure 4: Proportion of Streaming-Associated Releases",
    "Proportion (0.0 - 1.0)",
    "#9467bd",  # Matplotlib Purple
)

# ---------------------------------------------------------
# 4. EXTRA: SEASONALITY & CORRELATION CHECK
# ---------------------------------------------------------

# Seasonality Boxplots
# This is better than a line chart for showing the spread/variance per month
plt.figure(figsize=(14, 6))
sns.boxplot(x="month", y="movie_count", data=df, palette="Blues")
plt.title("Figure 5: Seasonality Check - Volume by Month", fontsize=16)
plt.xlabel("Month (1=Jan, 12=Dec)")
plt.ylabel("Movie Count")
plt.savefig("plot_seasonality.png")
plt.show()

# Correlation Heatmap
# Does streaming correlate with ROI? Do Ratings correlate with Volume?
plt.figure(figsize=(8, 6))
corr_matrix = df[["movie_count", "avg_rating", "avg_log_roi", "prop_streaming"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Figure 6: Variable Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("plot_correlation.png")
plt.show()

print("\nVisualization script finished.")

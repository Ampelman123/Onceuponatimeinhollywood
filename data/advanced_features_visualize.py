import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# --- CONFIGURATION ---
INPUT_FILE = "advanced_feature_time_series.csv"  # The file you just generated
SHOCK_DATE = pd.to_datetime("2020-03-01")
ADJUSTMENT_DATE = pd.to_datetime("2022-01-01")

# Load Data
df = pd.read_csv(INPUT_FILE)
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"])
    df.set_index("release_date", inplace=True)
else:
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)

# Filter out the "January Artifact" (Optional but recommended)
# We exclude months where count is suspiciously high relative to neighbors
# Simple heuristic: If count > 2x the rolling median, treat as outlier?
# For now, let's just plot what we have.

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 7)


def plot_creative_metric(column, title, ylabel, color):
    fig, ax = plt.subplots()

    # Raw Data
    ax.plot(
        df.index, df[column], color=color, alpha=0.3, linewidth=1.5, label="Monthly"
    )

    # Trend
    rolling = df[column].rolling(window=6, center=True).mean()
    ax.plot(df.index, rolling, color=color, linewidth=3, label="6-Month Trend")

    # Interventions
    ax.axvline(x=SHOCK_DATE, color="red", linestyle="--", label="COVID Shock")
    ax.axvline(x=ADJUSTMENT_DATE, color="green", linestyle="--", label="Adjustment")

    # Shading
    ax.axvspan(df.index.min(), SHOCK_DATE, alpha=0.05, color="gray")
    ax.axvspan(SHOCK_DATE, ADJUSTMENT_DATE, alpha=0.1, color="red")
    ax.axvspan(ADJUSTMENT_DATE, df.index.max(), alpha=0.1, color="green")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{column}.png")
    plt.show()


# 1. Gini Coefficient (The "Inequality" Plot)
plot_creative_metric(
    "revenue_gini",
    "Figure 7: Revenue Concentration (Gini Coefficient)",
    "Inequality (0=Equal, 1=Monopoly)",
    "#d62728",
)  # Red

# 2. Shannon Entropy (The "Diversity" Plot)
plot_creative_metric(
    "genre_entropy",
    "Figure 8: Cultural Diversity (Genre Entropy)",
    "Shannon Entropy (Bits)",
    "#9467bd",
)  # Purple

# 3. Sequel Proportion (The "Risk Aversion" Plot)
plot_creative_metric(
    "prop_sequels",
    "Figure 9: Reliance on Established IP (Sequels)",
    "Proportion of Movies",
    "#17becf",
)  # Teal

"""
Time-series visualizations (monthly) with era shading.
Uses tueplots for publication-friendly styling.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from tueplots import bundles
except ImportError as exc:
    raise ImportError(
        "tueplots is required for this script. Install with: pip install tueplots"
    ) from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "movies_with_release_type.csv")

METRICS = [
    ("runtime", "Runtime", "minutes", 1),
    ("budget", "Budget", "million USD", 1e6),
    ("revenue", "Revenue", "million USD", 1e6),
    ("popularity", "Popularity", "score", 1),
    ("vote_count", "Engagement", "votes", 1),
]


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update(
        {
            "text.usetex": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "lines.linewidth": 2.2,
        }
    )


def load_monthly_data():
    csv.field_size_limit(1000000)
    theatrical_by_month = defaultdict(list)
    streaming_by_month = defaultdict(list)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("release_date", "")
            if not date:
                continue

            try:
                year = int(date.split("-")[0])
                month = int(date.split("-")[1])
                year_month = f"{year}-{month:02d}"
            except ValueError:
                continue

            if year < 2017 or year > 2024:
                continue

            strategy = row.get("release_strategy", "").lower()
            if strategy not in ["theatrical", "streaming"]:
                continue

            film = {}
            try:
                runtime = int(row.get("runtime", 0))
                if 40 <= runtime <= 300:
                    film["runtime"] = runtime
            except ValueError:
                pass

            try:
                budget = int(row.get("budget", 0))
                if budget > 0:
                    film["budget"] = budget
            except ValueError:
                pass

            try:
                revenue = int(row.get("revenue", 0))
                if revenue > 0:
                    film["revenue"] = revenue
            except ValueError:
                pass

            try:
                popularity = float(row.get("popularity", 0))
                if popularity > 0:
                    film["popularity"] = popularity
            except ValueError:
                pass

            try:
                vote_count = int(row.get("vote_count", 0))
                if vote_count > 0:
                    film["vote_count"] = vote_count
            except ValueError:
                pass

            if strategy == "theatrical":
                theatrical_by_month[year_month].append(film)
            else:
                streaming_by_month[year_month].append(film)

    return theatrical_by_month, streaming_by_month


def rolling_avg(data, window=3):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = [d for d in data[start : i + 1] if not np.isnan(d)]
        smoothed.append(np.mean(window_data) if window_data else np.nan)
    return smoothed


def linear_regression(x, y):
    if len(x) < 2:
        return None, None
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
    if denominator == 0:
        return None, None
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def add_period_trends(ax, dates, data, color, alpha=0.45):
    pre_end = datetime(2019, 12, 31)
    covid_end = datetime(2021, 12, 31)

    def plot_segment(mask):
        idx = [i for i, m in enumerate(mask) if m]
        y_vals = [data[i] for i in idx if not np.isnan(data[i])]
        x_vals = [i for i in idx if not np.isnan(data[i])]
        if len(y_vals) > 1:
            slope, intercept = linear_regression(x_vals, y_vals)
            if slope is not None:
                trend = [slope * i + intercept for i in idx]
                ax.plot([dates[i] for i in idx], trend, color=color, linestyle="--",
                        linewidth=1.8, alpha=alpha, zorder=2)

    plot_segment([d <= pre_end for d in dates])
    plot_segment([(pre_end < d <= covid_end) for d in dates])
    plot_segment([d > covid_end for d in dates])


def create_metric_chart(metric_key, metric_name, unit, divisor, filename):
    theatrical_by_month, streaming_by_month = load_monthly_data()
    all_months = sorted(set(theatrical_by_month.keys()) | set(streaming_by_month.keys()))
    dates = [datetime.strptime(ym, "%Y-%m") for ym in all_months]

    t_vals = []
    s_vals = []
    for ym in all_months:
        t_list = [f[metric_key] / divisor for f in theatrical_by_month[ym] if metric_key in f]
        s_list = [f[metric_key] / divisor for f in streaming_by_month[ym] if metric_key in f]
        t_vals.append(np.mean(t_list) if t_list else np.nan)
        s_vals.append(np.mean(s_list) if s_list else np.nan)

    t_smooth = rolling_avg(t_vals, window=3)
    s_smooth = rolling_avg(s_vals, window=3)

    fig, ax = plt.subplots(figsize=(14, 4.8), facecolor="white")

    t_color = "#1e40af"
    s_color = "#b91c1c"
    pre_end = datetime(2019, 12, 31)
    covid_end = datetime(2021, 12, 31)

    ax.axvspan(dates[0], pre_end, alpha=0.08, color="#6b7280", zorder=0)
    ax.axvspan(pre_end, covid_end, alpha=0.12, color="#f59e0b", zorder=0)
    ax.axvspan(covid_end, dates[-1], alpha=0.08, color="#10b981", zorder=0)

    ax.plot(dates, t_smooth, color=t_color, label="Theatrical", zorder=3)
    ax.plot(dates, s_smooth, color=s_color, label="Streaming", zorder=3)

    add_period_trends(ax, dates, t_smooth, t_color)
    add_period_trends(ax, dates, s_smooth, s_color)

    ax.set_title(f"{metric_name} Evolution (Monthly, 2017-2024)")
    ax.set_ylabel(f"{metric_name} ({unit})")
    ax.set_xlabel("Time Period")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#d1d5db")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Created: {filename}")


def main():
    apply_plot_style()
    for metric_key, metric_name, unit, divisor in METRICS:
        out_name = f"{metric_key}_monthly.png"
        create_metric_chart(metric_key, metric_name, unit, divisor, out_name)


if __name__ == "__main__":
    main()

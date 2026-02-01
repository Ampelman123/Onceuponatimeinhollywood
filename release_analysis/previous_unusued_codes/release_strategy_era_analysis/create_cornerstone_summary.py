"""
Cornerstone figure: overall COVID-era shift and streaming vs theatrical trend.
Builds a composite index (normalized to each strategy's pre-COVID baseline)
from key metrics and plots it over time.
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
METRICS = ["runtime", "budget", "revenue", "popularity", "vote_count"]


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update(
        {
            "text.usetex": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
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


def build_metric_series(data_by_month, months):
    series = {m: [] for m in METRICS}
    for ym in months:
        rows = data_by_month.get(ym, [])
        for metric in METRICS:
            vals = [r[metric] for r in rows if metric in r]
            series[metric].append(np.mean(vals) if vals else np.nan)
    return series


def normalize_to_precovid(series, months):
    pre_months = [m for m in months if m <= "2019-12"]
    normed = {}
    for metric, values in series.items():
        pre_vals = [v for m, v in zip(months, values) if m in pre_months and not np.isnan(v)]
        baseline = np.mean(pre_vals) if pre_vals else np.nan
        if not baseline or np.isnan(baseline):
            normed[metric] = [np.nan for _ in values]
        else:
            normed[metric] = [v / baseline if not np.isnan(v) else np.nan for v in values]
    return normed


def composite_index(normed_series):
    comp = []
    for i in range(len(next(iter(normed_series.values())))):
        vals = [normed_series[m][i] for m in METRICS if not np.isnan(normed_series[m][i])]
        comp.append(np.mean(vals) if vals else np.nan)
    return comp


def create_cornerstone_chart(out_file="cornerstone_summary.png"):
    theatrical_by_month, streaming_by_month = load_monthly_data()
    all_months = sorted(set(theatrical_by_month.keys()) | set(streaming_by_month.keys()))
    dates = [datetime.strptime(ym, "%Y-%m") for ym in all_months]

    t_series = build_metric_series(theatrical_by_month, all_months)
    s_series = build_metric_series(streaming_by_month, all_months)

    t_norm = normalize_to_precovid(t_series, all_months)
    s_norm = normalize_to_precovid(s_series, all_months)

    t_comp = rolling_avg(composite_index(t_norm), window=3)
    s_comp = rolling_avg(composite_index(s_norm), window=3)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")

    t_color = "#1e40af"
    s_color = "#b91c1c"
    pre_end = datetime(2019, 12, 31)
    covid_end = datetime(2021, 12, 31)

    ax.axvspan(dates[0], pre_end, alpha=0.08, color="#6b7280", zorder=0)
    ax.axvspan(pre_end, covid_end, alpha=0.12, color="#f59e0b", zorder=0)
    ax.axvspan(covid_end, dates[-1], alpha=0.08, color="#10b981", zorder=0)

    ax.plot(dates, t_comp, color=t_color, label="Theatrical composite", zorder=3)
    ax.plot(dates, s_comp, color=s_color, label="Streaming composite", zorder=3)

    ax.set_title("Composite Industry Index (Normalized to Pre-COVID Baseline)")
    ax.set_ylabel("Index (Pre-COVID = 1.0)")
    ax.set_xlabel("Time Period")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#d1d5db")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Created: {out_file}")


if __name__ == "__main__":
    apply_plot_style()
    create_cornerstone_chart()

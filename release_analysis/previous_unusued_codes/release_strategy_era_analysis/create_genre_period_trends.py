"""
Genre trends across COVID periods (theatrical vs streaming), two-panel chart.
Matches the previous visual layout but uses tueplots styling.
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
except ImportError as exc:
    raise ImportError(
        "tueplots is required for this script. Install with: pip install tueplots"
    ) from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "movies_with_release_type.csv")
PERIODS = ["pre", "covid", "post"]
PERIOD_LABELS = ["Pre-COVID", "COVID-Shock", "Post-COVID"]


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
            "lines.linewidth": 2.0,
        }
    )


def parse_genres(genres_str):
    if not genres_str or genres_str == "[]":
        return []
    try:
        genres = json.loads(genres_str.replace("'", '"'))
        if isinstance(genres, list):
            return [g["name"] for g in genres if isinstance(g, dict) and "name" in g]
    except Exception:
        return []
    return []


def load_genre_period_counts():
    csv.field_size_limit(1000000)
    data = {
        "pre": {"theatrical": defaultdict(int), "streaming": defaultdict(int)},
        "covid": {"theatrical": defaultdict(int), "streaming": defaultdict(int)},
        "post": {"theatrical": defaultdict(int), "streaming": defaultdict(int)},
    }

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("release_date", "")
            if not date:
                continue
            try:
                year = int(date.split("-")[0])
            except ValueError:
                continue

            if year < 2017 or year > 2024:
                continue

            if year <= 2019:
                period = "pre"
            elif year <= 2021:
                period = "covid"
            else:
                period = "post"

            strategy = row.get("release_strategy", "").lower()
            if strategy not in ["theatrical", "streaming"]:
                continue

            genres = parse_genres(row.get("genres", ""))
            for genre in genres:
                data[period][strategy][genre] += 1

    return data


def top_genres(data, top_n=10):
    counts = defaultdict(int)
    for period in PERIODS:
        for strategy in ["theatrical", "streaming"]:
            for genre, cnt in data[period][strategy].items():
                counts[genre] += cnt
    return sorted(counts.keys(), key=lambda g: counts[g], reverse=True)[:top_n]


def compute_percentages(data, genres):
    theatrical = {g: [] for g in genres}
    streaming = {g: [] for g in genres}

    for period in PERIODS:
        t_total = sum(data[period]["theatrical"].values())
        s_total = sum(data[period]["streaming"].values())

        for genre in genres:
            t_pct = 100 * data[period]["theatrical"][genre] / t_total if t_total else 0
            s_pct = 100 * data[period]["streaming"][genre] / s_total if s_total else 0
            theatrical[genre].append(t_pct)
            streaming[genre].append(s_pct)

    return theatrical, streaming


def create_genre_trends_line_chart(out_file="genre_trends_tueplots.png", top_n=10):
    data = load_genre_period_counts()
    genres = top_genres(data, top_n=top_n)
    theatrical, streaming = compute_percentages(data, genres)

    colors = [
        "#1e40af", "#b91c1c", "#10b981", "#f59e0b",
        "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
        "#64748b", "#22c55e",
    ]

    x = np.arange(len(PERIODS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5), facecolor="white")
    fig.suptitle(
        "Genre Trends Across COVID Periods: Theatrical vs Streaming",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx, genre in enumerate(genres):
        ax1.plot(x, theatrical[genre], marker="o", label=genre, color=colors[idx], alpha=0.9)
        ax2.plot(x, streaming[genre], marker="o", label=genre, color=colors[idx], alpha=0.9)

    for ax, title in [(ax1, "Theatrical Releases"), (ax2, "Streaming Releases")]:
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_LABELS)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#d1d5db", ncol=2)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Created: {out_file}")


if __name__ == "__main__":
    apply_plot_style()
    create_genre_trends_line_chart()

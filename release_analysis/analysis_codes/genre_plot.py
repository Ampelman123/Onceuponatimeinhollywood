import ast
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tueplots import bundles
except ImportError:
    bundles = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_final.csv")

PERIODS = {
    "pre": (2017, 2019),
    "covid": (2020, 2021),
    "post": (2022, 2024),
}

GENRE_COLORS = {
    "Drama": "#1e40af",
    "Comedy": "#15803d",
    "Documentary": "#b91c1c",
    "Horror": "#9333ea",
    "Action": "#ea580c",
    "Thriller": "#0891b2",
    "Romance": "#db2777",
    "Crime": "#6b7280",
    "Adventure": "#0d9488",
    "Music": "#c026d3",
}


def apply_style():
    if bundles:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=2))
    plt.rcParams.update({
        "text.usetex": False,
    })


def parse_genres_cell(genres_str: str):

    if not genres_str:
        return []

    try:
        obj = ast.literal_eval(genres_str)
        if isinstance(obj, list):
            return [g.get("name") for g in obj if isinstance(g, dict) and "name" in g]
    except Exception:
        pass

    try:
        obj = json.loads(genres_str)
        if isinstance(obj, list):
            return [g.get("name") for g in obj if isinstance(g, dict) and "name" in g]
    except Exception:
        return []

    return []


def load_genre_data():
    csv.field_size_limit(1_000_000)

    data = {
        "theatrical": {"pre": defaultdict(int), "covid": defaultdict(int), "post": defaultdict(int)},
        "streaming": {"pre": defaultdict(int), "covid": defaultdict(int), "post": defaultdict(int)},
    }
    totals = {
        "theatrical": {"pre": 0, "covid": 0, "post": 0},
        "streaming": {"pre": 0, "covid": 0, "post": 0},
    }

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            date_str = row.get("release_date", "")
            if not date_str:
                continue

            try:
                year = int(date_str.split("-")[0])
            except Exception:
                continue

            period = None
            for p, (start, end) in PERIODS.items():
                if start <= year <= end:
                    period = p
                    break
            if not period:
                continue

            strategy = row.get("release_strategy", "").lower()
            if strategy not in ("theatrical", "streaming"):
                continue

            genre_names = parse_genres_cell(row.get("genres", ""))
            if not genre_names:
                continue

            for g in genre_names:
                data[strategy][period][g] += 1
            totals[strategy][period] += 1

    return data, totals


def calculate_percentages(data, totals):
    percentages = {
        "theatrical": {"pre": {}, "covid": {}, "post": {}},
        "streaming": {"pre": {}, "covid": {}, "post": {}},
    }

    for strategy in ("theatrical", "streaming"):
        for period in ("pre", "covid", "post"):
            total = totals[strategy][period]
            if total <= 0:
                continue
            for genre, count in data[strategy][period].items():
                percentages[strategy][period][genre] = (count / total) * 100.0

    return percentages


def get_top_genres(percentages, n=8):

    totals = defaultdict(float)
    for strategy in ("theatrical", "streaming"):
        for period in ("pre", "covid", "post"):
            for genre, pct in percentages[strategy][period].items():
                totals[genre] += pct

    return [g for g, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n]]


def create_genre_trends():
    data, totals = load_genre_data()

    percentages = calculate_percentages(data, totals)

    top_genres = get_top_genres(percentages, n=10)
    print(f"Top genres: {top_genres}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    periods = ["pre", "covid", "post"]
    period_labels = ["Pre-COVID", "COVID-Shock", "Post-COVID"]
    x_pos = np.arange(len(periods))

    colors = {}
    for i, genre in enumerate(top_genres):
        colors[genre] = GENRE_COLORS.get(genre, plt.cm.tab10(i % 10))

    ax1.set_title("Theatrical Releases", fontweight="bold", fontsize=13)
    for genre in top_genres:
        values = [percentages["theatrical"][p].get(genre, 0) for p in periods]
        ax1.plot(x_pos, values, marker="o", linewidth=2.5, markersize=7,
                 label=genre, color=colors[genre], alpha=0.85)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(period_labels)
    ax1.set_ylabel("% of Films", fontweight="bold")
    ax1.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_title("Streaming Releases", fontweight="bold", fontsize=13)
    for genre in top_genres:
        values = [percentages["streaming"][p].get(genre, 0) for p in periods]
        ax2.plot(x_pos, values, marker="s", linewidth=2.5, markersize=7,
                 label=genre, color=colors[genre], alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(period_labels)
    ax2.set_ylabel("% of Films", fontweight="bold")
    ax2.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               frameon=True, framealpha=0.95, edgecolor="#d1d5db",
               bbox_to_anchor=(0.5, 0.93), fontsize=9)

    fig.suptitle("Genre Trends Across COVID Periods: Theatrical vs Streaming",
                 fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = os.path.join(SCRIPT_DIR, "genre_trends.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def main():
    apply_style()
    create_genre_trends()


if __name__ == "__main__":
    main()

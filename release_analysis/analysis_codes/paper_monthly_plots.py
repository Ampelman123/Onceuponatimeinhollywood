import csv
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tueplots import bundles
    from tueplots.constants.color import rgb
except ImportError:
    bundles = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_final.csv")

PERIODS = {
    "pre": (datetime(2017, 1, 1), datetime(2019, 12, 31)),
    "covid": (datetime(2020, 1, 1), datetime(2021, 12, 31)),
    "post": (datetime(2022, 1, 1), datetime(2024, 12, 31)),
}


COLORS = {
    "theatrical": rgb.tue_blue,
    "streaming": rgb.tue_red,
}

PERIOD_BG = {
    "pre": rgb.tue_lightblue,   
    "covid": rgb.tue_mauve,      
    "post": rgb.tue_lightgreen,  
}


def apply_style():
    if bundles:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
    plt.rcParams.update({"text.usetex": False})


def load_monthly_lists():
    csv.field_size_limit(1_000_000)

    monthly_data = {
        "theatrical": defaultdict(lambda: {"runtime": [], "budget": [], "revenue": [], "popularity": []}),
        "streaming": defaultdict(lambda: {"runtime": [], "budget": [], "revenue": [], "popularity": []}),
    }

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            date_str = row.get("release_date", "")
            if not date_str:
                continue
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue

            if date.year < 2017 or date.year > 2024:
                continue

            strategy = row.get("release_strategy", "").lower()
            if strategy not in ("theatrical", "streaming"):
                continue

            month_key = date.strftime("%Y-%m")

            # runtime
            try:
                runtime = float(row.get("runtime", 0) or 0)
                if runtime > 0:
                    monthly_data[strategy][month_key]["runtime"].append(runtime)
            except Exception:
                pass

            # budget in millions
            try:
                budget = float(row.get("budget", 0) or 0)
                if budget > 0:
                    monthly_data[strategy][month_key]["budget"].append(budget / 1e6)
            except Exception:
                pass

            # revenue in millions
            try:
                revenue = float(row.get("revenue", 0) or 0)
                if revenue > 0:
                    monthly_data[strategy][month_key]["revenue"].append(revenue / 1e6)
            except Exception:
                pass

            # popularity
            try:
                pop = float(row.get("popularity", 0) or 0)
                if pop > 0:
                    monthly_data[strategy][month_key]["popularity"].append(pop)
            except Exception:
                pass

    return monthly_data


def monthly_means(monthly_lists):

    means = {"theatrical": {}, "streaming": {}}
    metrics = ["runtime", "budget", "revenue", "popularity"]

    for strategy in ("theatrical", "streaming"):
        months_sorted = sorted(monthly_lists[strategy].keys())
        for metric in metrics:
            xs, ys = [], []
            for mk in months_sorted:
                vals = monthly_lists[strategy][mk][metric]
                if vals:
                    xs.append(datetime.strptime(mk, "%Y-%m"))
                    ys.append(float(np.mean(vals)))
            means[strategy][metric] = {"months": xs, "values": ys}

    return means


def moving_average(values, window=3):
    if window <= 1 or len(values) < window:
        return values
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = values[lo:hi].mean()
    return out.tolist()


def add_period_background(ax):
    for pname, (start, end) in PERIODS.items():
        ax.axvspan(start, end, alpha=0.12, color=PERIOD_BG[pname], zorder=0)


def add_period_trends(ax, months, values, color):

    months = np.asarray(months)
    values = np.asarray(values, dtype=float)

    for pname, (start, end) in PERIODS.items():
        mask = (months >= start) & (months <= end)
        if mask.sum() < 4:
            continue
        x_period = months[mask]
        y_period = values[mask]

        t = np.arange(len(y_period))
        slope, intercept = np.polyfit(t, y_period, 1)
        trend = slope * t + intercept

        ax.plot(x_period, trend, linestyle="--", color=color, alpha=0.5, linewidth=1.3)


def create_metric_plot(means, metric, title, ylabel, out_prefix, smooth_window=3):
    fig, ax = plt.subplots()
    add_period_background(ax)

    for strategy in ("theatrical", "streaming"):
        x = means[strategy][metric]["months"]
        y = means[strategy][metric]["values"]
        if not y:
            continue

        y_s = moving_average(y, window=smooth_window)

        # raw (faint)
        ax.plot(x, y, color=COLORS[strategy], alpha=0.25, linewidth=1.0,
                label=f"{strategy.capitalize()} (raw)")

        # smoothed (strong)
        ax.plot(x, y_s, color=COLORS[strategy], alpha=0.9, linewidth=2.2,
                label=f"{strategy.capitalize()} (smoothed)")

        add_period_trends(ax, x, y_s, COLORS[strategy])

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()

    pdf_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Created: {pdf_path}")
    print(f"✓ Created: {png_path}")


def main():
    apply_style()

    lists_ = load_monthly_lists()
    means = monthly_means(lists_)

    specs = [
        ("runtime", "Runtime Evolution", "Runtime (minutes)", "runtime_monthly"),
        ("budget", "Budget Evolution", "Budget (million USD)", "budget_monthly"),
        ("revenue", "Revenue Evolution", "Revenue (million USD)", "revenue_monthly"),
        ("popularity", "Popularity Evolution", "Popularity", "popularity_monthly"),
    ]

    for metric, title, ylabel, out_prefix in specs:
        create_metric_plot(means, metric, title, ylabel, out_prefix, smooth_window=3)

if __name__ == "__main__":
    main()

import csv
import os
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots.constants.color import rgb
from tueplots import bundles

import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.plotting_style import set_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../../data/dataset_final.csv")

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
    set_style(column="half", nrows=1, ncols=1)

    # Ensure all text is the same size as requested (ICML bundle consistency)
    base_size = plt.rcParams["font.size"]
    plt.rcParams.update(
        {
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
            "text.usetex": False,
            "figure.constrained_layout.use": False,  # Disable to allow manual subplots_adjust
        }
    )


# ... (skipping unchanged code)


def create_metric_plot(
    means,
    metric,
    title,
    ylabel,
    out_prefix,
    smooth_window=3,
    show_legend=False,
    nrows=1,
):
    # Calculate figsize based on specific nrows for this plot
    rc_params = bundles.icml2024(column="half", nrows=nrows, ncols=1)
    figsize = rc_params["figure.figsize"]

    # Explicitly disable layout engine to respect subplots_adjust
    fig, ax = plt.subplots(figsize=figsize, layout=None)
    add_period_background(ax)

    # ... (skipping plotting code)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        # Place legend under the plot
        # Adjusted for ncol=1 so it fits in width
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=1,
            frameon=False,
            fontsize=plt.rcParams["font.size"],
        )

    # Manual margin adjustment for consistent Axes dimensions
    # Standard Physical Height = H (nrows=1)

    # Target Geometry (Physical units relative to Standard H):
    # Left Margin: 0.23H
    # Right Margin: 0.02H (Pos: 0.98)
    # Top Margin: 0.05H (Pos: 0.95)
    # Axes Height: 0.75H (Standard Bottom: 0.20)

    left_std = 0.23
    right_std = 0.98
    top_std = 0.95
    bottom_std = 0.20

    if nrows > 1.1:
        # Runtime case (nrows=1.3)
        scale = nrows
        # Maintain physical top margin
        top_new = 1.0 - ((1.0 - top_std) / scale)

        # Maintain physical axes height
        axes_height_new = (top_std - bottom_std) / scale

        # Bottom is remainder
        bottom_new = top_new - axes_height_new

        fig.subplots_adjust(
            left=left_std, right=right_std, top=top_new, bottom=bottom_new
        )
    else:
        # Standard case
        fig.subplots_adjust(
            left=left_std, right=right_std, top=top_std, bottom=bottom_std
        )

    pdf_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path)  # bbox_inches="tight" removed to enforce strict geometry
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def load_monthly_lists():
    csv.field_size_limit(1_000_000)

    monthly_data = {
        "theatrical": defaultdict(
            lambda: {"runtime": [], "budget": [], "revenue": [], "popularity": []}
        ),
        "streaming": defaultdict(
            lambda: {"runtime": [], "budget": [], "revenue": [], "popularity": []}
        ),
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


def create_metric_plot(
    means,
    metric,
    title,
    ylabel,
    out_prefix,
    smooth_window=3,
    show_legend=False,
    nrows=1,
):
    # Calculate figsize based on specific nrows for this plot (preserving "half" column width)
    # We use the bundle to get the correct aspect ratio/height
    rc_params = bundles.icml2024(column="half", nrows=nrows, ncols=1)
    figsize = rc_params["figure.figsize"]

    fig, ax = plt.subplots(figsize=figsize)
    add_period_background(ax)

    for strategy in ("theatrical", "streaming"):
        x = means[strategy][metric]["months"]
        y = means[strategy][metric]["values"]
        if not y:
            continue

        y_s = moving_average(y, window=smooth_window)

        # raw (faint)
        ax.plot(
            x,
            y,
            color=COLORS[strategy],
            alpha=0.25,
            linewidth=1.0,
            label=f"{strategy.capitalize()} (raw)",
        )

        # smoothed (strong)
        ax.plot(
            x,
            y_s,
            color=COLORS[strategy],
            alpha=0.9,
            linewidth=2.2,
            label=f"{strategy.capitalize()} (smoothed)",
        )

        add_period_trends(ax, x, y_s, COLORS[strategy])

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        # Place legend under the plot
        # Move left to align with Y-label (Runtime minutes) and tight vertical fit
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.16, -0.25),
            ncol=2,
            frameon=False,
            fontsize=plt.rcParams["font.size"],
        )

    # Manual margin adjustment for consistent Axes dimensions
    # Standard Physical Height = H (nrows=1)
    # We want physical top margin to be constant = (1 - top_std) * 1.0 = 0.05
    # We want physical axes height to be constant = (top_std - bottom_std) * 1.0 = 0.75

    left_std = 0.17
    right_std = 0.98
    top_std = 0.95
    bottom_std = 0.20

    scale = nrows

    # New top fraction: 1 - (physical_margin / scale)
    top_new = 1.0 - ((1.0 - top_std) / scale)

    # New axes height fraction: physical_height / scale
    axes_height_fraction = (top_std - bottom_std) / scale

    bottom_new = top_new - axes_height_fraction

    fig.subplots_adjust(left=left_std, right=right_std, top=top_new, bottom=bottom_new)

    # fig.tight_layout() # Disabled to respect manual margins

    pdf_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
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
        is_runtime = metric == "runtime"
        create_metric_plot(
            means,
            metric,
            title,
            ylabel,
            out_prefix,
            smooth_window=3,
            show_legend=is_runtime,
            nrows=1.25 if is_runtime else 1.0,
        )


if __name__ == "__main__":
    main()

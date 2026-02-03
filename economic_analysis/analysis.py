#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import math
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from tueplots import bundles
    from tueplots.constants.color import rgb
except Exception:
    bundles = None
    rgb = None


def apply_style(column: str = "full", nrows: int = 1, ncols: int = 1, dpi: int = 140) -> None:
    if bundles is not None:
        plt.rcParams.update(bundles.icml2024(column=column, nrows=nrows, ncols=ncols, usetex=False))
    plt.rcParams.update(
        {
            "text.usetex": False,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def tu_colors() -> Dict[str, Tuple[float, float, float]]:
    if rgb is None:
        return {
            "pre": (0.35, 0.65, 0.35),
            "covid": (0.75, 0.25, 0.25),
            "post": (0.25, 0.45, 0.75),
        }
    return {
        "pre": tuple(rgb.tue_green),
        "covid": tuple(rgb.tue_red),
        "post": tuple(rgb.tue_blue),
    }


def save_fig(fig: plt.Figure, out_dir: str, name: str, dpi: int = 200) -> None:
    os.makedirs(out_dir, exist_ok=True)
    svg_path = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {svg_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-parametric comparisons + plots (ICML/tueplots style).")
    p.add_argument("--csv", default="movies_with_release_type.csv", help="Path to CSV file.")
    p.add_argument("--start-year", type=int, default=2015, help="Start year (inclusive).")
    p.add_argument("--end-year", type=int, default=2024, help="End year (inclusive).")
    p.add_argument("--min-budget", type=float, default=1_000_000, help="Minimum budget.")
    p.add_argument("--min-revenue", type=float, default=1_000_000, help="Minimum revenue.")
    p.add_argument("--min-runtime", type=float, default=40, help="Minimum runtime (minutes).")
    p.add_argument("--n-perm", type=int, default=10_000, help="Permutation iterations.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--plots-dir", default="plots", help="Output directory for plots.")
    p.add_argument("--dpi", type=int, default=200, help="Savefig dpi (png).")
    p.add_argument("--bundle-column", choices=["half", "full"], default="full", help="tueplots column size.")
    p.add_argument("--top-n", type=int, default=10, help="Top N genres for radar/heatmap.")
    p.add_argument("--statistic", choices=["median", "trimmed_mean"], default="median")
    p.add_argument("--trim", type=float, default=0.2)
    return p.parse_args()


def covid_period_label(release_date: pd.Series) -> pd.Series:
    conditions = [
        release_date <= "2020-03-10",
        (release_date >= "2020-03-11") & (release_date <= "2021-12-31"),
        release_date >= "2022-01-01",
    ]
    choices = ["Before COVID", "During COVID", "After COVID"]
    return np.select(conditions, choices, default="Unknown")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def trimmed_mean(values: np.ndarray, trim: float) -> float:
    if values.size == 0:
        return np.nan
    values = np.sort(values)
    k = int(math.floor(trim * values.size))
    if k * 2 >= values.size:
        return np.nan
    return float(values[k:-k].mean())


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    statistic: str,
    n_perm: int,
    rng: np.random.Generator,
    trim: float,
) -> Tuple[float, float]:
    if statistic == "median":
        stat_fn = np.median
    else:
        stat_fn = lambda x: trimmed_mean(x, trim)

    observed = float(stat_fn(a) - stat_fn(b))
    combined = np.concatenate([a, b])
    n_a = a.size
    count = 0

    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_stat = float(stat_fn(perm_a) - stat_fn(perm_b))
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    n1 = a.size
    n2 = b.size
    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()

    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    _, counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    tie_correction = 1 - tie_sum / (combined.size**3 - combined.size)

    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 * tie_correction)
    if sigma == 0:
        return float(u), np.nan

    z = (u - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(u), float(p)


def describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "iqr": float(np.subtract(*np.percentile(values, [75, 25]))) if values.size else np.nan,
    }


def run_pairwise_tests(
    data: pd.DataFrame,
    value_col: str,
    statistic: str,
    n_perm: int,
    rng: np.random.Generator,
    trim: float,
) -> pd.DataFrame:
    groups = ["Before COVID", "During COVID", "After COVID"]
    results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            a = data.loc[data["period"] == g1, value_col].dropna().to_numpy()
            b = data.loc[data["period"] == g2, value_col].dropna().to_numpy()
            if a.size == 0 or b.size == 0:
                continue
            u_stat, u_p = mann_whitney_u(a, b)
            perm_stat, perm_p = permutation_test(a, b, statistic, n_perm, rng, trim)

            results.append(
                {
                    "metric": value_col,
                    "group_a": g1,
                    "group_b": g2,
                    "n_a": a.size,
                    "n_b": b.size,
                    "mann_whitney_u": u_stat,
                    "mann_whitney_p": u_p,
                    "perm_stat": perm_stat,
                    "perm_p": perm_p,
                }
            )
    return pd.DataFrame(results)


def extract_first_genre(value: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        data = ast.literal_eval(value)
    except Exception:
        return None
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "name" in first:
            return first["name"]
    return None


def plot_combined_boxplots_radar(
    df: pd.DataFrame, out_dir: str, top_n: int, dpi: int, column: str
) -> None:
    apply_style(column=column, nrows=1, ncols=3, dpi=140)

    periods = ["Before COVID", "During COVID", "After COVID"]
    labels = ["Pre-COVID", "During", "After"]
    box_metrics = ["budget", "revenue"]

    fig = plt.figure(constrained_layout=True)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height * 1.75, forward=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    ax_budget = fig.add_subplot(gs[0, 0])
    ax_revenue = fig.add_subplot(gs[0, 1])
    ax_radar = fig.add_subplot(gs[0, 2], polar=True)

    for ax, metric in zip([ax_budget, ax_revenue], box_metrics):
        data = [df.loc[df["period"] == p, metric].dropna().to_numpy() for p in periods]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Period")
        ax.set_ylabel(metric)

        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (log scale)")

        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_roi_radar_centered(df, top_n=top_n, ax=ax_radar)

    fig.suptitle("Budget, Revenue, and ROI by COVID Period", y=1.02)
    save_fig(fig, out_dir, "period_boxplots_radar", dpi=dpi)


def add_period_indicator_strip(fig: plt.Figure, labels, colors, y_offset: float = -0.06) -> None:
    """
    A dedicated strip below the radar (no overlap with genre labels).
    Uses drawn rectangles (not unicode bullets), so it will not “scramble”.
    """
    ax_leg = fig.add_axes([0.18, y_offset, 0.64, 0.045])
    ax_leg.set_axis_off()

    xs = [0.00, 0.36, 0.70]
    sw = 0.06
    sh = 0.38
    y = 0.31

    label_size = plt.rcParams.get("legend.fontsize", plt.rcParams.get("font.size", 10))
    for x, lab, col in zip(xs, labels, colors):
        ax_leg.add_patch(Rectangle((x, y), sw, sh, facecolor=col, edgecolor=col, linewidth=1.0))
        ax_leg.text(
            x + sw + 0.03,
            y + sh / 2,
            lab,
            va="center",
            ha="left",
            fontsize=label_size,
        )


def plot_roi_radar_centered(
    df: pd.DataFrame,
    out_dir: str | None = None,
    top_n: int = 10,
    dpi: int = 200,
    column: str = "full",
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> None:
    c = tu_colors()

    df = df.copy()
    df["primary_genre"] = df["genres"].apply(extract_first_genre)
    df = df[df["primary_genre"].notna()].copy()

    top_genres = df["primary_genre"].value_counts().head(top_n).index.tolist()
    df = df[df["primary_genre"].isin(top_genres)].copy()

    periods = ["Before COVID", "During COVID", "After COVID"]
    period_titles = ["Pre-COVID", "COVID", "Post-COVID"]
    period_colors = {
        "Before COVID": c["pre"],
        "During COVID": c["covid"],
        "After COVID": c["post"],
    }

    angles = np.linspace(0, 2 * np.pi, len(top_genres), endpoint=False).tolist()
    angles += angles[:1]

    own_fig = False
    if ax is None:
        apply_style(column=column, nrows=1, ncols=1, dpi=140)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, polar=True)
        # More bottom margin so the lowest genre label (e.g. ADVENTURE) has room,
        # and the indicator strip sits BELOW it.
        fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.34)
        own_fig = True
    else:
        fig = ax.figure

    for period, title in zip(periods, period_titles):
        slice_df = df[df["period"] == period]
        medians = (
            slice_df.groupby("primary_genre")["roi"]
            .median()
            .reindex(top_genres)
            .fillna(0.0)
        )
        vals = medians.to_numpy(dtype=float).tolist()
        vals += vals[:1]

        ax.plot(angles, vals, linewidth=2.0, color=period_colors[period], label=title)
        ax.fill(angles, vals, color=period_colors[period], alpha=0.14)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    tick_size = plt.rcParams.get("xtick.labelsize", plt.rcParams.get("font.size", 10))
    ax.set_xticklabels([g.upper() for g in top_genres], fontsize=tick_size)
    ax.tick_params(axis="x", pad=14)

    ax.grid(alpha=0.6, color="#333333", linestyle="--", linewidth=0.8)
    title_size = plt.rcParams.get("axes.titlesize", plt.rcParams.get("font.size", 10))
    ax.set_title("Median ROI by Genre × Period", fontsize=title_size, pad=10)

    # Colorful bordered boxes for genre labels
    genre_box_cols = plt.cm.tab10(np.linspace(0, 1, len(top_genres)))
    for tick, col in zip(ax.get_xticklabels(), genre_box_cols):
        tick.set_bbox(
            dict(
                edgecolor=col,
                facecolor=(1, 1, 1, 0.0),
                boxstyle="round,pad=0.25",
                linewidth=1.4,
            )
        )

    if show_legend:
        legend_size = plt.rcParams.get("legend.fontsize", plt.rcParams.get("font.size", 8))
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=3,
            frameon=False,
            fontsize=legend_size,
        )

    # Dedicated strip BELOW the plot — no overlap with genre labels.
    if own_fig and not show_legend:
        add_period_indicator_strip(
            fig,
            labels=period_titles,
            colors=[period_colors[p] for p in periods],
        )

    if own_fig and out_dir is not None:
        save_fig(fig, out_dir, "roi_radar_by_period", dpi=dpi)


def plot_roi_heatmap(df: pd.DataFrame, out_dir: str, top_n: int, dpi: int, column: str) -> None:
    apply_style(column=column, nrows=1, ncols=1, dpi=140)

    df = df.copy()
    df["primary_genre"] = df["genres"].apply(extract_first_genre)
    df = df[df["primary_genre"].notna()].copy()

    top_genres = df["primary_genre"].value_counts().head(top_n).index.tolist()
    df = df[df["primary_genre"].isin(top_genres)].copy()

    periods = ["Before COVID", "During COVID", "After COVID"]

    heat = (
        df.pivot_table(
            index="primary_genre",
            columns="period",
            values="roi",
            aggfunc="median",
        )
        .reindex(index=top_genres, columns=periods)
    )

    fig, ax = plt.subplots(constrained_layout=True)

    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(range(len(periods)))
    tick_size = plt.rcParams.get("xtick.labelsize", plt.rcParams.get("font.size", 10))
    ax.set_xticklabels(["Pre", "During", "Post"], fontsize=tick_size)
    ax.set_yticks(range(len(top_genres)))
    ytick_size = plt.rcParams.get("ytick.labelsize", plt.rcParams.get("font.size", 10))
    ax.set_yticklabels(top_genres, fontsize=ytick_size)
    title_size = plt.rcParams.get("axes.titlesize", plt.rcParams.get("font.size", 10))
    ax.set_title("Median ROI by Genre × Period", fontsize=title_size)

    cbar = fig.colorbar(im, ax=ax)
    label_size = plt.rcParams.get("axes.labelsize", plt.rcParams.get("font.size", 10))
    cbar.set_label("Median ROI", fontsize=label_size)

    save_fig(fig, out_dir, "roi_heatmap_genre_period", dpi=dpi)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[df["release_date"].dt.year.between(args.start_year, args.end_year)]

    df["budget"] = clean_numeric(df["budget"])
    df["revenue"] = clean_numeric(df["revenue"])
    df["runtime"] = clean_numeric(df["runtime"])

    df = df[
        (df["budget"] >= args.min_budget)
        & (df["revenue"] >= args.min_revenue)
        & (df["runtime"] >= args.min_runtime)
    ].copy()

    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]
    df["period"] = covid_period_label(df["release_date"])
    df = df[df["period"] != "Unknown"].copy()

    print("Sample sizes by period:")
    print(df["period"].value_counts().to_string())
    print()

    metrics = ["budget", "revenue", "roi"]
    for metric in metrics:
        print(f"Descriptives for {metric}:")
        desc = (
            df.groupby("period")[metric]
            .apply(lambda s: describe(s.dropna().to_numpy()))
            .apply(pd.Series)
        )
        print(desc.to_string())
        print()

    all_results = []
    for metric in metrics:
        results = run_pairwise_tests(df, metric, args.statistic, args.n_perm, rng, args.trim)
        all_results.append(results)

    final = pd.concat(all_results, ignore_index=True)
    print("Pairwise tests:")
    print(final.to_string(index=False))
    print()

    print(f"Writing plots to: {args.plots_dir}")
    plot_combined_boxplots_radar(
        df,
        args.plots_dir,
        top_n=args.top_n,
        dpi=args.dpi,
        column=args.bundle_column,
    )
    plot_roi_heatmap(df, args.plots_dir, top_n=args.top_n, dpi=args.dpi, column=args.bundle_column)


if __name__ == "__main__":
    main()

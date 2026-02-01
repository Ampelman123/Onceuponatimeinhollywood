import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
except ImportError:
    bundles = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_final.csv")

PERIODS = {"pre": (2017, 2019), "covid": (2020, 2021), "post": (2022, 2024)}
PERIOD_COLORS = {"pre": "#1e40af", "covid": "#b91c1c", "post": "#15803d"}
LABEL_MAP = {
    "pre": "Pre-COVID (2017–2019)",
    "covid": "COVID-Shock (2020–2021)",
    "post": "Post-COVID (2022–2024)",
}


def apply_style():
    if bundles:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
    plt.rcParams.update({"text.usetex": False})


def load_runtime_data_combined():
    csv.field_size_limit(1_000_000)
    data = {"pre": [], "covid": [], "post": []}

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
            for p, (a, b) in PERIODS.items():
                if a <= year <= b:
                    period = p
                    break
            if period is None:
                continue

            try:
                rt = float(row.get("runtime", 0) or 0)
                if rt > 0:
                    data[period].append(rt)
            except Exception:
                pass

    return data


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def dkw_epsilon(n, alpha=0.05):
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def create_dkw_plot(out_prefix="runtime_dkw"):
    data = load_runtime_data_combined()

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    medians = {}
    for period in ("pre", "covid", "post"):
        vals = np.asarray(data[period], dtype=float)
        if len(vals) == 0:
            continue

        x, y = ecdf(vals)
        eps = dkw_epsilon(len(vals), alpha=0.05)
        c = PERIOD_COLORS[period]

        ax.plot(x, y, color=c, linewidth=2.6, alpha=0.95,
                label=f"{LABEL_MAP[period]} (n={len(vals):,})")
        ax.fill_between(
            x,
            np.maximum(y - eps, 0.0),
            np.minimum(y + eps, 1.0),
            color=c,
            alpha=0.18,
            linewidth=0,
        )

        med = float(np.median(vals))
        medians[period] = med
        ax.axvline(med, color=c, linestyle=":", alpha=0.7, linewidth=1.8)

    if "pre" in medians and "covid" in medians:
        pre_med = medians["pre"]
        covid_med = medians["covid"]
        diff = covid_med - pre_med

        y_level = 0.55
        ax.annotate(
            "",
            xy=(covid_med, y_level),
            xytext=(pre_med, y_level),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
        )
        ax.text(
            (covid_med + pre_med) / 2.0,
            y_level + 0.03,
            f"{diff:+.1f} min",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray"),
        )

    ax.set_xlabel("Runtime (minutes)", fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontweight="bold")
    ax.set_title("Runtime Distribution: DKW",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(40, 180)

    fig.tight_layout()

    pdf_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.pdf")
    png_path = os.path.join(SCRIPT_DIR, f"{out_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main():
    apply_style()
    create_dkw_plot()


if __name__ == "__main__":
    main()

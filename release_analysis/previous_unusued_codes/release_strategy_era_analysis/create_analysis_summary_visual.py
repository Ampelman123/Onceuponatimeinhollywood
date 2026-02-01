"""
Visual summary of analysis_results.txt
P-value table across era comparisons (Theatrical vs Streaming).
"""

import re
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


INPUT_PATH = "analysis_results.txt"
OUTPUT_PATH = "analysis_summary_pvalues.png"

METRICS = ["Runtime", "Budget", "Revenue", "Popularity", "Audience Engagement"]
COMPARISONS = ["pre vs covid", "covid vs post", "pre vs post"]


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update(
        {
            "text.usetex": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def parse_results(text):
    data = {"THEATRICAL": {}, "STREAMING": {}}
    current_metric = None
    current_section = None

    metric_map = {
        "Runtime (minutes)": "Runtime",
        "Budget (USD)": "Budget",
        "Revenue (USD)": "Revenue",
        "Popularity Score": "Popularity",
        "Audience Engagement": "Audience Engagement",
    }

    for line in text.splitlines():
        line = line.strip()
        if line in metric_map:
            current_metric = metric_map[line]
            data["THEATRICAL"][current_metric] = {}
            data["STREAMING"][current_metric] = {}
            continue
        if line == "THEATRICAL:":
            current_section = "THEATRICAL"
            continue
        if line == "STREAMING:":
            current_section = "STREAMING"
            continue

        match = re.match(r"(pre vs covid|covid vs post|pre vs post): .* p=([0-9.]+) d=([-0-9.]+) .*", line)
        if match and current_metric and current_section:
            comp = match.group(1)
            p_val = float(match.group(2))
            d_val = float(match.group(3))
            data[current_section][current_metric][comp] = {"p": p_val, "d": d_val}

    return data


def build_matrix(data, section):
    matrix = []
    for metric in METRICS:
        row = []
        for comp in COMPARISONS:
            entry = data[section].get(metric, {}).get(comp, {"p": np.nan, "d": 0.0})
            row.append(entry["p"])
        matrix.append(row)
    return np.array(matrix)


def create_pvalue_table():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    data = parse_results(text)
    t_matrix = build_matrix(data, "THEATRICAL")
    s_matrix = build_matrix(data, "STREAMING")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5), facecolor="white", constrained_layout=True
    )
    fig.suptitle("Era Comparisons Summary: P-values", y=0.98)

    col_labels = ["Pre–Covid", "Covid–Post", "Pre–Post"]

    def render_table(ax, title, matrix):
        ax.axis("off")
        table = ax.table(
            cellText=[[f"{v:.4f}" if not np.isnan(v) else "-" for v in row] for row in matrix],
            rowLabels=METRICS,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.scale(1, 1.4)
        ax.set_title(title, pad=10)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            if row == 0:
                cell.set_text_props(weight="bold")
            if col == -1:
                cell.set_text_props(weight="bold")
            if row > 0 and col >= 0:
                try:
                    p_val = float(cell.get_text().get_text())
                    if p_val < 0.05:
                        cell.set_facecolor("#fee2e2")
                except ValueError:
                    pass

    render_table(ax1, "Theatrical", t_matrix)
    render_table(ax2, "Streaming", s_matrix)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    apply_plot_style()
    create_pvalue_table()

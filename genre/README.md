# Genre Analysis

This directory contains the code and analysis for the **Genre Shifts Project**.

## Directory Structure

*   `plots/`: Python scripts to generate all plots for the paper/report.
*   `scripts/`: Analysis scripts for data processing and metric calculation (Epps-Singleton, Gini, etc.).
*   `../data/`: Shared data directory (raw and processed data).
*   `../utils/`: Shared utilities (Plotting styles, etc.).

## How to Run

It is recommended to run scripts using `uv` or from the root directory to ensure imports work correctly.

### Generating Plots

```bash
uv run python genre/plots/genre_slopegraph.py
uv run python genre/plots/genre_financial_health_trend.py
uv run python genre/plots/genre_market_health_trend.py
uv run python genre/plots/genre_franchise_saturation_trend.py
uv run python genre/plots/genre_integrated_financial_franchise.py
uv run python genre/plots/genre_drift_plot.py
uv run python genre/plots/plot_efficiency_fragility.py
uv run python genre/plots/genre_correlation_matrix.py
```

### Running Analysis

```bash
uv run python genre/scripts/calculate_epps_singleton.py
uv run python genre/scripts/update_metrics.py
uv run python genre/scripts/genre_gini.py
uv run python genre/scripts/genre_elasticity.py
uv run python genre/scripts/generate_genre_metrics.py
```

All outputs will be saved to `plots/` or `data/` accordingly.

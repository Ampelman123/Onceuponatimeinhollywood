#!/usr/bin/env python3
"""
Production scale visualizations using tueplots styling.
Creates separate images for each analysis aspect.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from tueplots import bundles
except ImportError as exc:
    raise ImportError("tueplots required. Install with: pip install tueplots") from exc

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'new_tmdb_movies_master.jsonl')
ALPHA = 0.05
N_BOOTSTRAP = 10000

# Color scheme matching release_strategy_era_analysis
COLORS = {
    'pre': '#1e40af',      # Deep blue
    'covid': '#b91c1c',    # Deep red
    'post': '#047857'      # Deep green
}

# Temporal regimes
PRE_COVID = (datetime(2017, 1, 1), datetime(2019, 12, 31))
COVID_SHOCK = (datetime(2020, 1, 1), datetime(2021, 12, 31))
POST_COVID = (datetime(2022, 1, 1), datetime(2024, 12, 31))

np.random.seed(42)


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'lines.linewidth': 2.2,
    })


def load_production_data():
    """Load production countries data."""
    data = {
        'pre_covid': [],
        'covid_shock': [],
        'post_covid': []
    }
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                date_str = record.get('release_date', '')
                prod_countries = record.get('production_countries', [])
                
                if not date_str:
                    continue
                
                date = datetime.strptime(date_str, '%Y-%m-%d')
                n_countries = len(prod_countries) if prod_countries else 0
                
                if PRE_COVID[0] <= date <= PRE_COVID[1]:
                    data['pre_covid'].append(n_countries)
                elif COVID_SHOCK[0] <= date <= COVID_SHOCK[1]:
                    data['covid_shock'].append(n_countries)
                elif POST_COVID[0] <= date <= POST_COVID[1]:
                    data['post_covid'].append(n_countries)
            except:
                pass
    
    return {k: np.array(v) for k, v in data.items()}


def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap confidence intervals for proportions."""
    proportions = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        proportions.append(np.mean(sample > 1))  # International (2+ countries)
    
    prop_est = np.mean(proportions)
    prop_ci_lower = np.percentile(proportions, 2.5)
    prop_ci_upper = np.percentile(proportions, 97.5)
    
    return prop_est, prop_ci_lower, prop_ci_upper


def create_international_coproduction_chart(data):
    """Chart 1: International co-production rates with confidence intervals."""
    
    periods = ['pre_covid', 'covid_shock', 'post_covid']
    period_labels = ['Pre-COVID\n(2017-2019)', 'COVID-Shock\n(2020-2021)', 'Post-COVID\n(2022-2024)']
    colors_list = [COLORS['pre'], COLORS['covid'], COLORS['post']]
    
    # Calculate proportions and CIs
    stats = [bootstrap_ci(data[p]) for p in periods]
    props = [s[0] * 100 for s in stats]
    ci_lowers = [s[1] * 100 for s in stats]
    ci_uppers = [s[2] * 100 for s in stats]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    x_pos = np.arange(len(periods))
    errors = [[p - l for p, l in zip(props, ci_lowers)],
              [u - p for p, u in zip(props, ci_uppers)]]
    
    bars = ax.bar(x_pos, props, color=colors_list, alpha=0.85, edgecolor='#374151', linewidth=1.5)
    ax.errorbar(x_pos, props, yerr=errors, fmt='none', color='#374151', 
                linewidth=2, capsize=8, capthick=2, zorder=10)
    
    # Add value labels
    for bar, val in zip(bars, props):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('% of Films with 2+ Countries')
    ax.set_title('International Co-Production Rates Across COVID Periods')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(props) + 5)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'international_coproduction_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: international_coproduction_tueplots.png")


def create_production_distribution_chart(data):
    """Chart 2: Distribution of number of production countries."""
    
    periods = ['pre_covid', 'covid_shock', 'post_covid']
    period_labels = ['Pre-COVID', 'COVID-Shock', 'Post-COVID']
    colors_list = [COLORS['pre'], COLORS['covid'], COLORS['post']]
    
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    
    max_countries = max(max(data[p]) for p in periods)
    bins = np.arange(0, min(max_countries + 1, 8))
    
    for period, label, color in zip(periods, period_labels, colors_list):
        counts, _ = np.histogram(data[period], bins=bins)
        percentages = counts / len(data[period]) * 100
        
        ax.plot(bins[:-1], percentages, marker='o', linewidth=2.5, markersize=8,
                label=f'{label} (n={len(data[period]):,})', color=color, alpha=0.9)
    
    ax.set_xlabel('Number of Production Countries')
    ax.set_ylabel('% of Films')
    ax.set_title('Distribution of Production Countries Per Film')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    ax.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(bins[:-1])
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'production_distribution_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: production_distribution_tueplots.png")


def create_domestic_vs_international_chart(data):
    """Chart 3: Stacked bar showing domestic vs international split."""
    
    periods = ['pre_covid', 'covid_shock', 'post_covid']
    period_labels = ['Pre-COVID\n(2017-2019)', 'COVID-Shock\n(2020-2021)', 'Post-COVID\n(2022-2024)']
    colors_list = [COLORS['pre'], COLORS['covid'], COLORS['post']]
    
    domestic_counts = [np.sum(data[p] <= 1) / len(data[p]) * 100 for p in periods]
    international_counts = [np.sum(data[p] > 1) / len(data[p]) * 100 for p in periods]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    x_pos = np.arange(len(periods))
    width = 0.6
    
    # Create stacked bars with borders
    bars1 = ax.bar(x_pos, domestic_counts, width, color='#e5e7eb', 
                   edgecolor='#6b7280', linewidth=1.5, label='Domestic (0-1 country)')
    bars2 = ax.bar(x_pos, international_counts, width, bottom=domestic_counts,
                   color='#fbbf24', edgecolor='#f59e0b', linewidth=1.5,
                   label='International (2+ countries)')
    
    # Add percentage labels
    for i, (dom, intl) in enumerate(zip(domestic_counts, international_counts)):
        # International label
        ax.text(i, dom + intl/2, f'{intl:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#78350f')
        # Domestic label
        ax.text(i, dom/2, f'{dom:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#4b5563')
    
    ax.set_ylabel('% of Films')
    ax.set_title('Domestic vs International Production Scale')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'domestic_vs_international_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: domestic_vs_international_tueplots.png")


def create_change_from_baseline_chart(data):
    """Chart 4: Change in international co-production from pre-COVID baseline."""
    
    periods = ['pre_covid', 'covid_shock', 'post_covid']
    period_labels = ['Pre-COVID\n(2017-2019)', 'COVID-Shock\n(2020-2021)', 'Post-COVID\n(2022-2024)']
    colors_list = [COLORS['pre'], COLORS['covid'], COLORS['post']]
    
    # Calculate proportions
    prop_pre = np.mean(data['pre_covid'] > 1) * 100
    prop_covid = np.mean(data['covid_shock'] > 1) * 100
    prop_post = np.mean(data['post_covid'] > 1) * 100
    
    changes = [0, prop_covid - prop_pre, prop_post - prop_pre]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    x_pos = np.arange(len(periods))
    bars = ax.bar(x_pos, changes, color=colors_list, alpha=0.85, 
                  edgecolor='#374151', linewidth=1.5)
    
    # Zero line
    ax.axhline(0, color='#6b7280', linestyle='--', linewidth=1.5, zorder=2)
    
    # Value labels
    for i, (bar, val) in enumerate(zip(bars, changes)):
        if val != 0:
            ax.text(bar.get_x() + bar.get_width()/2., 
                   val + 0.4 if val > 0 else val - 0.4,
                   f'{val:+.1f}pp', ha='center', 
                   va='bottom' if val > 0 else 'top',
                   fontsize=11, fontweight='bold')
    
    # Sample sizes at bottom
    samples = [len(data[p]) for p in periods]
    for i, n in enumerate(samples):
        ax.text(i, ax.get_ylim()[0] * 0.85, f'n = {n:,}', ha='center', 
                fontsize=9, style='italic', color='#6b7280')
    
    ax.set_ylabel('Change from Pre-COVID Baseline (pp)')
    ax.set_title('International Co-Production: Change from Pre-COVID')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set ylim based on data range
    max_abs = max(abs(min(changes)), abs(max(changes)))
    ax.set_ylim(-max_abs - 2, max_abs + 2)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'baseline_change_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: baseline_change_tueplots.png")


def main():
    apply_plot_style()
    
    print("Loading production data...")
    data = load_production_data()
    
    print("Creating visualizations...")
    create_international_coproduction_chart(data)
    create_production_distribution_chart(data)
    create_domestic_vs_international_chart(data)
    create_change_from_baseline_chart(data)
    
    print("\n✓ All production scale visualizations created successfully!")


if __name__ == '__main__':
    main()

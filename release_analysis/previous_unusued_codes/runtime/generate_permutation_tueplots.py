"""
Permutation test visualization for runtime analysis.
Uses tueplots styling to match release_strategy_era_analysis aesthetics.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
except ImportError as exc:
    raise ImportError("tueplots required. Install with: pip install tueplots") from exc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'new_tmdb_movies_master.jsonl')

PRE_COVID_YEARS = [2017, 2018, 2019]
COVID_SHOCK_YEARS = [2020, 2021]
POST_COVID_YEARS = [2022, 2023, 2024]
MIN_RUNTIME, MAX_RUNTIME = 40, 300
PERM_ITER = 10000

# Color scheme matching release_strategy_era_analysis
HIST_COLOR = '#e5e7eb'      # Light gray for null distribution
EDGE_COLOR = '#9ca3af'      # Medium gray for edges
OBS_COLOR = '#b91c1c'       # Deep red for observed value
PVAL_BG = '#fef3c7'         # Light amber for p-value box

np.random.seed(42)


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'lines.linewidth': 2.0,
    })


def load_data():
    pre_covid, covid_shock, post_covid = [], [], []
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            try:
                r = json.loads(line)
                date = r.get('release_date', '')
                runtime = r.get('runtime')
                if not date or not runtime:
                    continue
                if not (MIN_RUNTIME <= runtime <= MAX_RUNTIME):
                    continue
                year = int(date.split('-')[0])
                if year in PRE_COVID_YEARS:
                    pre_covid.append(runtime)
                elif year in COVID_SHOCK_YEARS:
                    covid_shock.append(runtime)
                elif year in POST_COVID_YEARS:
                    post_covid.append(runtime)
            except:
                pass
    
    return np.array(pre_covid), np.array(covid_shock), np.array(post_covid)


def permutation_test_distribution(g1, g2, n_iter=PERM_ITER):
    obs = np.mean(g1) - np.mean(g2)
    pooled = np.concatenate([g1, g2])
    n1 = len(g1)
    null = []
    for _ in range(n_iter):
        np.random.shuffle(pooled)
        null.append(np.mean(pooled[:n1]) - np.mean(pooled[n1:]))
    return np.array(null), obs


def main():
    apply_plot_style()
    
    pre_covid, covid_shock, post_covid = load_data()
    
    # Run permutation tests
    null_pre_covid, obs_pre_covid = permutation_test_distribution(pre_covid, covid_shock)
    null_covid_post, obs_covid_post = permutation_test_distribution(covid_shock, post_covid)
    null_pre_post, obs_pre_post = permutation_test_distribution(pre_covid, post_covid)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='white')
    
    comparisons = [
        (null_pre_covid, obs_pre_covid, 'Pre-COVID vs COVID-Shock', axes[0]),
        (null_covid_post, obs_covid_post, 'COVID-Shock vs Post-COVID', axes[1]),
        (null_pre_post, obs_pre_post, 'Pre-COVID vs Post-COVID', axes[2])
    ]
    
    for null_dist, obs_diff, title, ax in comparisons:
        p_value = np.mean(np.abs(null_dist) >= np.abs(obs_diff))
        
        # Histogram of null distribution
        ax.hist(null_dist, bins=50, color=HIST_COLOR, edgecolor=EDGE_COLOR,
                linewidth=0.5, alpha=0.85, density=True, zorder=2)
        
        # Observed difference line
        ax.axvline(obs_diff, color=OBS_COLOR, linestyle='--', linewidth=2.5,
                   label=f'Observed: {obs_diff:+.2f} min', zorder=4)
        
        # Shade the rejection region
        xlim = ax.get_xlim()
        if obs_diff > 0:
            ax.axvspan(obs_diff, xlim[1], alpha=0.08, color=OBS_COLOR, zorder=1)
            ax.axvspan(xlim[0], -obs_diff, alpha=0.08, color=OBS_COLOR, zorder=1)
        else:
            ax.axvspan(obs_diff, xlim[0], alpha=0.08, color=OBS_COLOR, zorder=1)
            ax.axvspan(-obs_diff, xlim[1], alpha=0.08, color=OBS_COLOR, zorder=1)
        
        # P-value annotation
        ylim = ax.get_ylim()
        sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        ax.text(obs_diff, ylim[1] * 0.85, f'p = {p_value:.4f} {sig_marker}',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=PVAL_BG,
                         edgecolor='#d97706', linewidth=1, alpha=0.95),
                zorder=5)
        
        ax.set_xlabel('Mean Difference (minutes)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#d1d5db')
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Permutation Tests: Runtime Differences Across COVID Periods',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'permutation_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: permutation_tueplots.png")


if __name__ == '__main__':
    main()

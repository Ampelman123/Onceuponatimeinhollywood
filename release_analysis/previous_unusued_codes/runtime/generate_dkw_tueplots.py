"""
DKW Confidence Bands visualization for runtime analysis.
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
ALPHA = 0.05

# Color scheme matching release_strategy_era_analysis
COLORS = {
    'pre': '#1e40af',      # Deep blue
    'covid': '#b91c1c',    # Deep red
    'post': '#047857'      # Deep green
}

np.random.seed(42)


def apply_plot_style():
    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'lines.linewidth': 2.2,
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


def main():
    apply_plot_style()
    
    pre_covid, covid_shock, post_covid = load_data()
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    regimes = [
        (pre_covid, 'Pre-COVID (2017-2019)', COLORS['pre']),
        (covid_shock, 'COVID-Shock (2020-2021)', COLORS['covid']),
        (post_covid, 'Post-COVID (2022-2024)', COLORS['post'])
    ]
    
    # Plot ECDFs with DKW bands
    for data, label, color in regimes:
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n + 1) / n
        
        # DKW epsilon
        epsilon = np.sqrt(np.log(2 / ALPHA) / (2 * n))
        lower = np.clip(y - epsilon, 0, 1)
        upper = np.clip(y + epsilon, 0, 1)
        
        # Main ECDF line
        ax.step(x, y, where='post', label=f'{label} (n={n:,})', 
                color=color, linewidth=2.5, zorder=5)
        
        # DKW confidence band
        ax.fill_between(x, lower, upper, alpha=0.15, color=color, step='post', zorder=2)
    
    # Add median lines (subtle)
    medians = [
        (np.median(pre_covid), COLORS['pre']),
        (np.median(covid_shock), COLORS['covid']),
        (np.median(post_covid), COLORS['post'])
    ]
    
    for median, color in medians:
        ax.axvline(median, color=color, linestyle=':', alpha=0.5, linewidth=1.5, zorder=3)
    
    # Annotations for shifts
    pre_med = np.median(pre_covid)
    covid_med = np.median(covid_shock)
    post_med = np.median(post_covid)
    
    # COVID contraction annotation
    ax.annotate('', xy=(covid_med, 0.5), xytext=(pre_med, 0.5),
                arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.8))
    ax.text((pre_med + covid_med) / 2, 0.54, f'−{pre_med - covid_med:.1f} min',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#9ca3af', linewidth=1, alpha=0.95))
    
    # Post-COVID rebound annotation
    ax.annotate('', xy=(post_med, 0.65), xytext=(covid_med, 0.65),
                arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.8))
    ax.text((covid_med + post_med) / 2, 0.69, f'+{post_med - covid_med:.1f} min',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#9ca3af', linewidth=1, alpha=0.95))
    
    ax.set_xlabel('Runtime (minutes)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Runtime Distribution: DKW Confidence Bands Across COVID Periods')
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    ax.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(40, 180)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'runtime_dkw_tueplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created: runtime_dkw_tueplots.png")


if __name__ == '__main__':
    main()

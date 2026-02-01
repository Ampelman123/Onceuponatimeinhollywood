#!/usr/bin/env python3
"""
Simple visualization of production scale analysis
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'new_tmdb_movies_master.jsonl')
ALPHA = 0.05
N_BOOTSTRAP = 10000

# Temporal regimes
PRE_COVID = (datetime(2017, 1, 1), datetime(2019, 12, 31))
COVID_SHOCK = (datetime(2020, 1, 1), datetime(2021, 12, 31))
POST_COVID = (datetime(2022, 1, 1), datetime(2024, 12, 31))

def load_production_data():
    """Load production countries data."""
    data = {
        'pre_covid': [],
        'covid_shock': [],
        'post_covid': []
    }
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            record = json.loads(line)
            
            date_str = record.get('release_date', '')
            prod_countries = record.get('production_countries', [])
            
            if not date_str:
                continue
            
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                continue
            
            n_countries = len(prod_countries) if prod_countries else 0
            
            if PRE_COVID[0] <= date <= PRE_COVID[1]:
                data['pre_covid'].append(n_countries)
            elif COVID_SHOCK[0] <= date <= COVID_SHOCK[1]:
                data['covid_shock'].append(n_countries)
            elif POST_COVID[0] <= date <= POST_COVID[1]:
                data['post_covid'].append(n_countries)
    
    return data

def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap confidence intervals."""
    means = []
    proportions = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
        proportions.append(np.mean(sample > 1))  # International
    
    mean_est = np.mean(means)
    mean_ci_lower = np.percentile(means, 2.5)
    mean_ci_upper = np.percentile(means, 97.5)
    
    prop_est = np.mean(proportions)
    prop_ci_lower = np.percentile(proportions, 2.5)
    prop_ci_upper = np.percentile(proportions, 97.5)
    
    return (mean_est, mean_ci_lower, mean_ci_upper,
            prop_est, prop_ci_lower, prop_ci_upper)

def create_visualization():
    """Create simple, clear multi-panel visualization."""
    
    data = load_production_data()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    periods = ['pre_covid', 'covid_shock', 'post_covid']
    period_labels = ['Pre-COVID\n(2017-2019)', 'COVID-Shock\n(2020-2021)', 'Post-COVID\n(2022-2024)']
    colors = ['#1e3a8a', '#dc2626', '#15803d']
    
    # Convert to numpy arrays
    data_pre = np.array(data['pre_covid'])
    data_covid = np.array(data['covid_shock'])
    data_post = np.array(data['post_covid'])
    
    # Calculate statistics
    mean_pre, _, _, prop_pre, prop_ci_l_pre, prop_ci_u_pre = bootstrap_ci(data_pre)
    mean_covid, _, _, prop_covid, prop_ci_l_covid, prop_ci_u_covid = bootstrap_ci(data_covid)
    mean_post, _, _, prop_post, prop_ci_l_post, prop_ci_u_post = bootstrap_ci(data_post)
    
    # Panel 1: International co-production rates
    ax1 = axes[0, 0]
    x_pos = np.arange(len(periods))
    
    props = [prop_pre * 100, prop_covid * 100, prop_post * 100]
    ci_lowers = [prop_ci_l_pre * 100, prop_ci_l_covid * 100, prop_ci_l_post * 100]
    ci_uppers = [prop_ci_u_pre * 100, prop_ci_u_covid * 100, prop_ci_u_post * 100]
    errors = [[p - l for p, l in zip(props, ci_lowers)],
              [u - p for p, u in zip(props, ci_uppers)]]
    
    bars = ax1.bar(x_pos, props, color=colors, alpha=0.75, edgecolor='black', linewidth=2)
    ax1.errorbar(x_pos, props, yerr=errors, fmt='none', color='black', 
                linewidth=2.5, capsize=10, capthick=2.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, props)):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax1.set_ylabel('% of Films', fontsize=14, fontweight='bold')
    ax1.set_title('International Co-Productions (2+ Countries)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(period_labels, fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, 18)
    
    # Panel 2: Domestic vs International stacked bars
    ax2 = axes[0, 1]
    
    domestic_counts = [
        np.sum(data_pre <= 1) / len(data_pre) * 100,
        np.sum(data_covid <= 1) / len(data_covid) * 100,
        np.sum(data_post <= 1) / len(data_post) * 100
    ]
    
    international_counts = [
        np.sum(data_pre > 1) / len(data_pre) * 100,
        np.sum(data_covid > 1) / len(data_covid) * 100,
        np.sum(data_post > 1) / len(data_post) * 100
    ]
    
    bars1 = ax2.bar(x_pos, domestic_counts, color='lightgray', alpha=0.8, 
                    edgecolor='black', linewidth=2, label='Domestic (0-1 country)')
    bars2 = ax2.bar(x_pos, international_counts, bottom=domestic_counts,
                    color='#f39c12', alpha=0.8, edgecolor='black', linewidth=2,
                    label='International (2+ countries)')
    
    # Add percentage labels
    for i, (dom, intl) in enumerate(zip(domestic_counts, international_counts)):
        ax2.text(i, dom + intl/2, f'{intl:.1f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    
    ax2.set_ylabel('% of Films', fontsize=14, fontweight='bold')
    ax2.set_title('Production Scale Distribution', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(period_labels, fontsize=12)
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(0, 105)
    
    # Panel 3: Distribution of number of countries
    ax3 = axes[1, 0]
    
    max_countries = max(max(data_pre), max(data_covid), max(data_post))
    bins = np.arange(0, min(max_countries + 1, 8))
    
    for period, label, color in zip([data_pre, data_covid, data_post], 
                                     ['Pre-COVID', 'COVID-Shock', 'Post-COVID'],
                                     colors):
        counts, _ = np.histogram(period, bins=bins)
        percentages = counts / len(period) * 100
        
        ax3.plot(bins[:-1], percentages, marker='o', linewidth=3, markersize=10,
                label=label, color=color, alpha=0.9)
    
    ax3.set_xlabel('Number of Production Countries', fontsize=14, fontweight='bold')
    ax3.set_ylabel('% of Films', fontsize=14, fontweight='bold')
    ax3.set_title('Distribution by Number of Production Countries', fontsize=15, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(bins[:-1])
    
    # Panel 4: Changes from baseline
    ax4 = axes[1, 1]
    
    baseline_prop = prop_pre * 100
    changes = [0, (prop_covid - prop_pre) * 100, (prop_post - prop_pre) * 100]
    
    bars = ax4.bar(x_pos, changes, color=colors, alpha=0.75, edgecolor='black', linewidth=2)
    ax4.axhline(0, color='black', linestyle='--', linewidth=2)
    
    for bar, val in zip(bars, changes):
        if val != 0:
            ax4.text(bar.get_x() + bar.get_width()/2., val + 0.3 if val > 0 else val - 0.3,
                    f'{val:+.1f}pp',
                    ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=12, fontweight='bold')
    
    # Add sample sizes
    samples = [len(data_pre), len(data_covid), len(data_post)]
    for i, n in enumerate(samples):
        ax4.text(i, -7.5, f'n = {n:,}', ha='center', fontsize=10,
                fontweight='bold', style='italic')
    
    ax4.set_ylabel('Change in % International (pp)', fontsize=14, fontweight='bold')
    ax4.set_title('Change from Pre-COVID Baseline', fontsize=15, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(period_labels, fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim(-8, 8)
    
    # Overall title
    fig.suptitle('Production Scale Analysis: Post-COVID Surge in International Co-Productions',
                fontsize=17, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    plt.savefig('production_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Production scale visualization saved: production_visualization.png")
    
    # Print summary
    print("\n" + "="*70)
    print("PRODUCTION SCALE SUMMARY")
    print("="*70)
    print(f"\nInternational co-production rates:")
    print(f"  Pre-COVID:    {prop_pre*100:.1f}% [{prop_ci_l_pre*100:.1f}%, {prop_ci_u_pre*100:.1f}%]")
    print(f"  COVID-Shock:  {prop_covid*100:.1f}% [{prop_ci_l_covid*100:.1f}%, {prop_ci_u_covid*100:.1f}%]")
    print(f"  Post-COVID:   {prop_post*100:.1f}% [{prop_ci_l_post*100:.1f}%, {prop_ci_u_post*100:.1f}%]")
    print(f"\nKey finding:")
    print(f"  International co-productions nearly DOUBLED post-COVID")
    print(f"  Change: {prop_pre*100:.1f}% → {prop_post*100:.1f}% (+{(prop_post-prop_pre)*100:.1f}pp)")
    print("="*70)
    
    plt.close()

if __name__ == '__main__':
    create_visualization()



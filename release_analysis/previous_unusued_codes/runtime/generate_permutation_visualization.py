import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'new_tmdb_movies_master.jsonl')
PRE_COVID_YEARS = [2017, 2018, 2019]
COVID_SHOCK_YEARS = [2020, 2021]
POST_COVID_YEARS = [2022, 2023, 2024]
MIN_RUNTIME, MAX_RUNTIME = 40, 300
PERM_ITER = 10000
np.random.seed(42)

pre_covid, covid_shock, post_covid = [], [], []

with open(DATA_FILE, 'r') as f:
    for line in f:
        try:
            r = json.loads(line)
            date, runtime = r.get('release_date', ''), r.get('runtime')
            if not date or not runtime or not (MIN_RUNTIME <= runtime <= MAX_RUNTIME):
                continue
            year = int(date.split('-')[0])
            if year in PRE_COVID_YEARS: pre_covid.append(runtime)
            elif year in COVID_SHOCK_YEARS: covid_shock.append(runtime)
            elif year in POST_COVID_YEARS: post_covid.append(runtime)
        except: pass

pre_covid = np.array(pre_covid)
covid_shock = np.array(covid_shock)
post_covid = np.array(post_covid)

def permutation_test_distribution(g1, g2, n_iter=PERM_ITER):
    obs = np.mean(g1) - np.mean(g2)
    pooled = np.concatenate([g1, g2])
    n1 = len(g1)
    null = []
    for _ in range(n_iter):
        np.random.shuffle(pooled)
        null.append(np.mean(pooled[:n1]) - np.mean(pooled[n1:]))
    return np.array(null), obs

null_pre_covid, obs_pre_covid = permutation_test_distribution(pre_covid, covid_shock)
null_covid_post, obs_covid_post = permutation_test_distribution(covid_shock, post_covid)
null_pre_post, obs_pre_post = permutation_test_distribution(pre_covid, post_covid)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

comparisons = [
    (null_pre_covid, obs_pre_covid, 'Pre-COVID vs COVID-Shock', axes[0]),
    (null_covid_post, obs_covid_post, 'COVID-Shock vs Post-COVID', axes[1]),
    (null_pre_post, obs_pre_post, 'Pre-COVID vs Post-COVID', axes[2])
]

for null_dist, obs_diff, title, ax in comparisons:
    p_value = np.mean(np.abs(null_dist) >= np.abs(obs_diff))
    
    ax.hist(null_dist, bins=50, color='lightgray', edgecolor='black', 
            linewidth=0.5, alpha=0.7, density=True)
    
    ax.axvline(obs_diff, color='red', linestyle='--', linewidth=3, 
              label=f'Observed: {obs_diff:.2f} min')
    
    ylim = ax.get_ylim()
    ax.text(obs_diff, ylim[1]*0.85, f'p = {p_value:.4f}',
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                    edgecolor='black', linewidth=1.5, alpha=0.9))
    
    ax.set_xlabel('Mean Difference (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=9)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

fig.suptitle('Permutation Tests: Null Distributions vs Observed Differences', 
            fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('permutation_test.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: permutation_test.png")
plt.close()


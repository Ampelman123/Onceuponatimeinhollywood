import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'new_tmdb_movies_master.jsonl')
PRE_COVID_YEARS = [2017, 2018, 2019]
COVID_SHOCK_YEARS = [2020, 2021]
POST_COVID_YEARS = [2022, 2023, 2024]
MIN_RUNTIME, MAX_RUNTIME = 40, 300
ALPHA = 0.05
np.random.seed(42)

COLORS = {'pre': '#1e3a8a', 'covid': '#dc2626', 'post': '#15803d'}

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

fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

regimes = [
    (pre_covid, 'Pre-COVID (2017-2019)', COLORS['pre']),
    (covid_shock, 'COVID-Shock (2020-2021)', COLORS['covid']),
    (post_covid, 'Post-COVID (2022-2024)', COLORS['post'])
]

for data, label, color in regimes:
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    
    epsilon = np.sqrt(np.log(2 / ALPHA) / (2 * n))
    lower = np.clip(y - epsilon, 0, 1)
    upper = np.clip(y + epsilon, 0, 1)
    
    ax.step(x, y, where='post', label=label, color=color, linewidth=3, alpha=0.9, zorder=5)
    ax.step(x, y, where='post', color=color, linewidth=5, alpha=0.15, zorder=4)
    ax.fill_between(x, lower, upper, alpha=0.15, color=color, step='post', zorder=3)

pre_median = np.median(pre_covid)
covid_median = np.median(covid_shock)
post_median = np.median(post_covid)

ax.axvline(pre_median, color=COLORS['pre'], linestyle=':', alpha=0.4, linewidth=2, zorder=2)
ax.axvline(covid_median, color=COLORS['covid'], linestyle=':', alpha=0.4, linewidth=2, zorder=2)
ax.axvline(post_median, color=COLORS['post'], linestyle=':', alpha=0.4, linewidth=2, zorder=2)

y_arrow = 0.5
ax.annotate('', xy=(covid_median, y_arrow), xytext=(pre_median, y_arrow),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=2.5, shrinkA=0, shrinkB=0))
text1 = ax.text((pre_median + covid_median) / 2, y_arrow + 0.05,
               'Contraction\n−4.4 min', ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333', linewidth=1.5, alpha=0.95),
               zorder=10)
text1.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

y_arrow2 = 0.65
ax.annotate('', xy=(post_median, y_arrow2), xytext=(covid_median, y_arrow2),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=2.5, shrinkA=0, shrinkB=0))
text2 = ax.text((covid_median + post_median) / 2, y_arrow2 + 0.05,
               'Rebound\n+8.3 min', ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333', linewidth=1.5, alpha=0.95),
               zorder=10)
text2.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

ax.set_xlabel('Runtime (minutes)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
ax.set_title('Runtime Distribution Shifts Across COVID-19 Periods', fontsize=15, fontweight='bold', pad=15)

legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.95, shadow=True, fancybox=True)
legend.get_frame().set_linewidth(1.5)

ax.grid(True, alpha=0.25, linestyle='--', linewidth=1, zorder=1)
ax.set_xlim(40, 180)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', labelsize=11, width=1.5, length=5)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('RUNTIME_ANALYSIS.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: RUNTIME_ANALYSIS.png")
plt.close()


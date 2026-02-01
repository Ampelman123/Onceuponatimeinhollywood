import csv
import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'dataset_final.csv')

PERIODS = {
    "pre": (2017, 2019),
    "covid": (2020, 2021),
    "post": (2022, 2024),
}

np.random.seed(42)

def load_data():
    csv.field_size_limit(1000000)
    
    data = {
        'pre': {'company_count': [], 'country_count': [], 'has_multiple_countries': []},
        'covid': {'company_count': [], 'country_count': [], 'has_multiple_countries': []},
        'post': {'company_count': [], 'country_count': [], 'has_multiple_countries': []}
    }
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                date_str = row.get('release_date', '')
                if not date_str:
                    continue
                
                year = int(date_str.split('-')[0])
                
                period = None
                for p, (start, end) in PERIODS.items():
                    if start <= year <= end:
                        period = p
                        break
                
                if not period:
                    continue
                
                companies_str = row.get('production_companies', '')
                countries_str = row.get('production_countries', '')
                
                companies = []
                countries = []
                
                if companies_str:
                    try:
                        companies_str_fixed = companies_str.replace("'", '"')
                        companies = json.loads(companies_str_fixed)
                    except:
                        pass
                
                if countries_str:
                    try:
                        countries_str_fixed = countries_str.replace("'", '"')
                        countries = json.loads(countries_str_fixed)
                    except:
                        pass
                
                data[period]['company_count'].append(len(companies))
                data[period]['country_count'].append(len(countries))
                data[period]['has_multiple_countries'].append(1 if len(countries) > 1 else 0)
            except:
                pass
    
    return {k: {m: np.array(v) for m, v in metrics.items()} 
            for k, metrics in data.items()}

def bootstrap_ci_mean(data, n_bootstrap=10000, ci=95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    
    return np.mean(data), lower, upper

def bootstrap_ci_proportion(data, n_bootstrap=10000, ci=95):
    bootstrap_props = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_props.append(np.mean(sample) * 100)
    
    lower = np.percentile(bootstrap_props, (100 - ci) / 2)
    upper = np.percentile(bootstrap_props, 100 - (100 - ci) / 2)
    
    return np.mean(data) * 100, lower, upper

def permutation_test_proportion(group1, group2, n_permutations=10000):
    obs_diff = (np.sum(group1) / len(group1) - np.sum(group2) / len(group2)) * 100
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        p1 = np.sum(combined[:n1]) / n1
        p2 = np.sum(combined[n1:]) / len(combined[n1:])
        perm_diffs.append((p1 - p2) * 100)
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value

if __name__ == '__main__':
    print("Loading production data...")
    data = load_data()
    
    print("="*70)
    print("PRODUCTION SCALE & INTERNATIONALITY ANALYSIS")
    print("="*70)
    
    print("\n1. PRODUCTION COMPANIES PER FILM")
    print("-" * 70)
    for period in ['pre', 'covid', 'post']:
        cc = data[period]['company_count']
        mean_val, lower, upper = bootstrap_ci_mean(cc)
        print(f"{period.upper()}")
        print(f"  Sample size: {len(cc)}")
        print(f"  Mean: {mean_val:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])")
    
    print("\n2. PRODUCTION COUNTRIES PER FILM")
    print("-" * 70)
    for period in ['pre', 'covid', 'post']:
        cc = data[period]['country_count']
        mean_val, lower, upper = bootstrap_ci_mean(cc)
        print(f"{period.upper()}")
        print(f"  Sample size: {len(cc)}")
        print(f"  Mean: {mean_val:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])")
    
    print("\n3. INTERNATIONAL CO-PRODUCTIONS (% with multiple countries)")
    print("-" * 70)
    for period in ['pre', 'covid', 'post']:
        mc = data[period]['has_multiple_countries']
        prop, lower, upper = bootstrap_ci_proportion(mc)
        print(f"{period.upper()}")
        print(f"  {prop:.2f}% (95% CI: [{lower:.2f}%, {upper:.2f}%])")
    
    print("\n4. PERMUTATION TESTS: International Co-productions")
    print("-" * 70)
    
    diff_1, p_1 = permutation_test_proportion(
        data['pre']['has_multiple_countries'],
        data['covid']['has_multiple_countries']
    )
    print(f"Pre-COVID vs COVID-Shock:")
    print(f"  Difference: {diff_1:.2f} percentage points")
    print(f"  p-value: {p_1:.4f}")
    
    diff_2, p_2 = permutation_test_proportion(
        data['covid']['has_multiple_countries'],
        data['post']['has_multiple_countries']
    )
    print(f"\nCOVID-Shock vs Post-COVID:")
    print(f"  Difference: {diff_2:.2f} percentage points")
    print(f"  p-value: {p_2:.4f}")
    
    print("\n" + "="*70)

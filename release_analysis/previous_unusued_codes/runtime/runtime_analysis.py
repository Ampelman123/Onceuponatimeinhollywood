import csv
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', '..', 'dataset_final.csv')
MIN_RUNTIME = 40
MAX_RUNTIME = 250

PERIODS = {
    "pre": (2017, 2019),
    "covid": (2020, 2021),
    "post": (2022, 2024),
}

np.random.seed(42)

def load_data():
    csv.field_size_limit(1000000)
    data = {'pre': [], 'covid': [], 'post': []}
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                runtime = float(row.get('runtime', 0))
                date_str = row.get('release_date', '')
                
                if not runtime or not date_str:
                    continue
                if not (MIN_RUNTIME <= runtime <= MAX_RUNTIME):
                    continue
                
                year = int(date_str.split('-')[0])
                
                for period, (start, end) in PERIODS.items():
                    if start <= year <= end:
                        data[period].append(runtime)
                        break
            except:
                pass
    
    return {k: np.array(v) for k, v in data.items()}

def dkw_bands(data, alpha=0.05):
    n = len(data)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    
    sorted_data = np.sort(data)
    ecdf_values = np.arange(1, n + 1) / n
    
    lower_band = np.maximum(0, ecdf_values - epsilon)
    upper_band = np.minimum(1, ecdf_values + epsilon)
    
    return sorted_data, ecdf_values, lower_band, upper_band

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    
    return np.mean(data), lower, upper

def permutation_test(group1, group2, n_permutations=10000):
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        perm_diffs.append(perm_diff)
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

if __name__ == '__main__':
    print("Loading runtime data...")
    data = load_data()
    
    print("="*70)
    print("RUNTIME ANALYSIS: COVID-19 IMPACT ON FILM DURATION")
    print("="*70)
    
    print("\n1. DESCRIPTIVE STATISTICS")
    print("-" * 70)
    for period, runtimes in data.items():
        mean_val, lower, upper = bootstrap_ci(runtimes)
        print(f"{period.upper()}")
        print(f"  Sample size: {len(runtimes)}")
        print(f"  Mean: {mean_val:.2f} minutes")
        print(f"  95% CI: [{lower:.2f}, {upper:.2f}]")
        print(f"  Median: {np.median(runtimes):.2f} minutes")
        print(f"  Std Dev: {np.std(runtimes, ddof=1):.2f}")
    
    print("\n2. DKW CONFIDENCE BANDS (α=0.05)")
    print("-" * 70)
    for period, runtimes in data.items():
        sorted_data, ecdf, lower, upper = dkw_bands(runtimes)
        epsilon = np.sqrt(np.log(2.0 / 0.05) / (2.0 * len(runtimes)))
        print(f"{period.upper()}: ε = {epsilon:.4f}")
    
    print("\n3. PERMUTATION TESTS (10,000 iterations)")
    print("-" * 70)
    
    diff_1, p_1 = permutation_test(data['pre'], data['covid'])
    print(f"Pre-COVID vs COVID-Shock:")
    print(f"  Difference: {diff_1:.2f} minutes")
    print(f"  p-value: {p_1:.4f}")
    
    diff_2, p_2 = permutation_test(data['covid'], data['post'])
    print(f"\nCOVID-Shock vs Post-COVID:")
    print(f"  Difference: {diff_2:.2f} minutes")
    print(f"  p-value: {p_2:.4f}")
    
    diff_3, p_3 = permutation_test(data['pre'], data['post'])
    print(f"\nPre-COVID vs Post-COVID:")
    print(f"  Difference: {diff_3:.2f} minutes")
    print(f"  p-value: {p_3:.4f}")
    
    print("\n" + "="*70)

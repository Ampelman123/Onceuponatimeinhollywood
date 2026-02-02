

import csv
import os
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "..", "dataset_final.csv")
ALPHA = 0.05
N_BOOTSTRAP = 10000
N_PERMUTATION = 10000

PERIODS = {
    "pre": (2017, 2019),
    "covid": (2020, 2021),
    "post": (2022, 2024),
}

METRICS = [
    ("runtime", "Runtime (minutes)"),
    ("budget", "Budget (USD)"),
    ("revenue", "Revenue (USD)"),
    ("popularity", "Popularity Score"),
]

CORR_METRICS = ["runtime", "budget", "revenue", "popularity"]


def load_and_filter_data():
    csv.field_size_limit(1000000)
    data = {"theatrical": defaultdict(list), "streaming": defaultdict(list)}

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("release_date", "")
            if not date:
                continue

            try:
                year = int(date.split("-")[0])
            except ValueError:
                continue

            period = None
            for p, (start, end) in PERIODS.items():
                if start <= year <= end:
                    period = p
                    break
            if not period:
                continue

            strategy = row.get("release_strategy", "").lower()
            if strategy not in ["theatrical", "streaming"]:
                continue

            film = {"year": year}

            try:
                runtime = int(row.get("runtime", 0))
                if 40 <= runtime <= 300:
                    film["runtime"] = runtime
            except ValueError:
                pass

            try:
                budget = int(row.get("budget", 0))
                if budget > 0:
                    film["budget"] = budget
            except ValueError:
                pass

            try:
                revenue = int(row.get("revenue", 0))
                if revenue > 0:
                    film["revenue"] = revenue
            except ValueError:
                pass

            try:
                popularity = float(row.get("popularity", 0))
                if popularity > 0:
                    film["popularity"] = popularity
            except ValueError:
                pass


            data[strategy][period].append(film)

    return data


def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP, ci=95):
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    lower_p = (100 - ci) / 2
    upper_p = 100 - lower_p
    return np.mean(values), np.percentile(means, lower_p), np.percentile(means, upper_p)


def permutation_test(group1, group2, n_permutations=N_PERMUTATION):
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    observed = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:n1]) - np.mean(combined[n1:]))

    perm_diffs = np.array(perm_diffs)
    return np.mean(np.abs(perm_diffs) >= np.abs(observed))


def dkw_epsilon(values, alpha=ALPHA):
    n = len(values)
    if n == 0:
        return np.nan
    return np.sqrt(np.log(2 / alpha) / (2 * n))


def cohens_d(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def period_pairs():
    return [("pre", "covid"), ("covid", "post"), ("pre", "post")]


def extract_values(data, strategy, period, metric):
    return np.array([f[metric] for f in data[strategy][period] if metric in f])


def correlation_matrix(rows, metrics):
    if len(rows) == 0:
        return None

    values = []
    for metric in metrics:
        series = [r[metric] for r in rows if metric in r]
        values.append(series)

    min_len = min(len(series) for series in values)
    if min_len < 10:
        return None

    aligned = np.array([series[:min_len] for series in values], dtype=float)
    return np.corrcoef(aligned)


def analyze_metric(data, metric_key, metric_name):
    results = defaultdict(dict)

    for strategy in ["theatrical", "streaming"]:
        for period in ["pre", "covid", "post"]:
            values = extract_values(data, strategy, period, metric_key)
            if len(values) == 0:
                continue

            mean, ci_low, ci_high = bootstrap_mean_ci(values)
            results[strategy][period] = {
                "n": len(values),
                "mean": mean,
                "ci_95": (ci_low, ci_high),
                "median": np.median(values),
                "std": np.std(values),
                "dkw_epsilon": dkw_epsilon(values),
            }

        for p1, p2 in period_pairs():
            v1 = extract_values(data, strategy, p1, metric_key)
            v2 = extract_values(data, strategy, p2, metric_key)
            if len(v1) == 0 or len(v2) == 0:
                continue

            p_value = permutation_test(v1, v2)
            results[strategy][f"{p1}_vs_{p2}"] = {
                "mean_diff": np.mean(v1) - np.mean(v2),
                "p_value": p_value,
                "significant": p_value < ALPHA,
                "cohens_d": cohens_d(v1, v2),
            }

    return {"metric": metric_name, "results": results}


def analyze_correlations(data):
    corr_results = {}
    for strategy in ["theatrical", "streaming"]:
        corr_results[strategy] = {}
        for period in ["pre", "covid", "post"]:
            rows = data[strategy][period]
            corr = correlation_matrix(rows, CORR_METRICS)
            corr_results[strategy][period] = corr
    return corr_results


def write_results(metric_results, corr_results, out_path="analysis_results.txt"):
    with open(out_path, "w") as f:
        f.write("ERA COMPARISON ANALYSIS (2017-2024)\n")
        f.write("=" * 72 + "\n\n")
        f.write("Focus: Pre vs COVID vs Post within each release strategy.\n")
        f.write("Methods: Bootstrap CI, Permutation tests, DKW bands, Cohen's d.\n\n")

        for item in metric_results:
            f.write("-" * 72 + "\n")
            f.write(f"{item['metric']}\n")
            f.write("-" * 72 + "\n")

            for strategy in ["theatrical", "streaming"]:
                f.write(f"\n{strategy.upper()}:\n")
                for period in ["pre", "covid", "post"]:
                    if period in item["results"][strategy]:
                        r = item["results"][strategy][period]
                        f.write(
                            f"  {period}: n={r['n']} mean={r['mean']:.2f} "
                            f"CI95=[{r['ci_95'][0]:.2f}, {r['ci_95'][1]:.2f}] "
                            f"DKW ε={r['dkw_epsilon']:.4f}\n"
                        )

                for p1, p2 in period_pairs():
                    key = f"{p1}_vs_{p2}"
                    if key in item["results"][strategy]:
                        r = item["results"][strategy][key]
                        f.write(
                            f"  {p1} vs {p2}: Δmean={r['mean_diff']:.2f} "
                            f"p={r['p_value']:.4f} d={r['cohens_d']:.3f} "
                            f"sig={'YES' if r['significant'] else 'NO'}\n"
                        )
                f.write("\n")

        f.write("\n" + "=" * 72 + "\n")
        f.write("WITHIN-PERIOD FEATURE CORRELATIONS (Pearson)\n")
        f.write("=" * 72 + "\n\n")
        f.write("Metrics order: runtime, budget, revenue, popularity\n\n")
        for strategy in ["theatrical", "streaming"]:
            f.write(f"{strategy.upper()}:\n")
            for period in ["pre", "covid", "post"]:
                corr = corr_results[strategy][period]
                if corr is None:
                    f.write(f"  {period}: insufficient data\n")
                    continue
                f.write(f"  {period}:\n")
                for row in corr:
                    f.write("    " + " ".join(f"{v: .2f}" for v in row) + "\n")
            f.write("\n")


def main():
    print("Loading data...")
    data = load_and_filter_data()
    print("Running era comparisons...")

    metric_results = []
    for metric_key, metric_name in METRICS:
        metric_results.append(analyze_metric(data, metric_key, metric_name))

    print("Computing correlations...")
    corr_results = analyze_correlations(data)

    write_results(metric_results, corr_results)
    print("✓ Results saved to analysis_results.txt")


if __name__ == "__main__":
    np.random.seed(42)
    main()

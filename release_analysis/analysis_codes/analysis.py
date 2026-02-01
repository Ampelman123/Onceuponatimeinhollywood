import csv
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_final.csv")

PERIODS = {
    "pre": (2017, 2019),
    "covid": (2020, 2021),
    "post": (2022, 2024),
}

ALPHA = 0.05
N_BOOT = 10000
N_PERM = 10000
SEED = 42

rng = np.random.default_rng(SEED)


def safe_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "" or x.lower() in {"nan", "none", "null"}:
            return None
        return float(x)
    except:
        return None


def year_from_date(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except:
        return None


def which_period(year):
    if year is None:
        return None
    for p, (a, b) in PERIODS.items():
        if a <= year <= b:
            return p
    return None


def load_data():
    csv.field_size_limit(1000000)

    data = {
        "theatrical": {"pre": [], "covid": [], "post": []},
        "streaming": {"pre": [], "covid": [], "post": []},
    }

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            strategy = (row.get("release_strategy", "") or "").strip().lower()
            if strategy not in {"theatrical", "streaming"}:
                continue

            year = year_from_date(row.get("release_date", ""))
            period = which_period(year)
            if period is None:
                continue

            runtime = safe_float(row.get("runtime"))
            popularity = safe_float(row.get("popularity"))
            budget = safe_float(row.get("budget"))
            revenue = safe_float(row.get("revenue"))

            if runtime is None or runtime <= 0:
                continue

            d = {
                "runtime": runtime,
                "popularity": popularity if popularity is not None else np.nan,
                "budget": budget if (budget is not None and budget > 0) else None,
                "revenue": revenue if (revenue is not None and revenue > 0) else None,
            }

            data[strategy][period].append(d)

    return data


def bootstrap_ci(values, B=N_BOOT, ci=95, stat="mean"):
    n = len(values)
    if n == 0:
        return np.nan, (np.nan, np.nan)

    values = np.asarray(values, dtype=float)

    if stat == "mean":
        point = float(np.mean(values))
        stats = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            stats[b] = np.mean(values[idx])
    elif stat == "median":
        point = float(np.median(values))
        stats = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            stats[b] = np.median(values[idx])
    else:
        return np.nan, (np.nan, np.nan)

    lo = (100 - ci) / 2
    hi = 100 - lo
    return point, (float(np.percentile(stats, lo)), float(np.percentile(stats, hi)))


def permutation_test_mean_diff(x, y, B=N_PERM):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return np.nan

    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    N = n1 + n2

    count = 0
    for _ in range(B):
        perm = rng.permutation(N)
        g1 = pooled[perm[:n1]]
        g2 = pooled[perm[n1:]]
        diff = float(np.mean(g1) - np.mean(g2))
        if abs(diff) >= abs(obs):
            count += 1

    p = (count + 1) / (B + 1)
    return float(p)


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1 = len(x)
    n2 = len(y)
    if n1 < 2 or n2 < 2:
        return np.nan

    m1 = np.mean(x)
    m2 = np.mean(y)
    s1 = np.std(x, ddof=1)
    s2 = np.std(y, ddof=1)

    sp = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan

    return float((m1 - m2) / sp)


def bonferroni_adjust(p_values):
    m = len(p_values)
    adj = []
    for p in p_values:
        if p is None or np.isnan(p):
            adj.append(np.nan)
        else:
            adj.append(float(min(1.0, p * m)))
    return adj


def get_values(data, strategy, period, key, transform=None):
    rows = data[strategy][period]
    total = len(rows)

    vals = []
    if key in {"budget", "revenue"}:
        for r in rows:
            v = r.get(key, None)
            if v is None:
                continue
            if transform == "log1p":
                vals.append(np.log1p(v))
            else:
                vals.append(v)
    else:
        for r in rows:
            v = r.get(key, None)
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            vals.append(v)

    return total, np.asarray(vals, dtype=float)


def analyze_one_metric(data, metric_name, key, transforms):
    out = {"metric": metric_name, "key": key, "blocks": []}

    for transform in transforms:
        block = {"transform": transform, "by_strategy": {}}

        for strategy in ["theatrical", "streaming"]:
            per_stats = {}
            for period in ["pre", "covid", "post"]:
                n_total, vals = get_values(data, strategy, period, key, transform)
                mean_pt, (mean_lo, mean_hi) = bootstrap_ci(vals, stat="mean")
                med_pt, (med_lo, med_hi) = bootstrap_ci(vals, stat="median")

                per_stats[period] = {
                    "n_total": n_total,
                    "n_used": int(len(vals)),
                    "mean": mean_pt,
                    "ci_mean": (mean_lo, mean_hi),
                    "median": med_pt,
                    "ci_median": (med_lo, med_hi),
                    "vals": vals,
                }

            comps = [("pre", "covid"), ("covid", "post"), ("pre", "post")]
            raw_ps = []
            comp_rows = []

            for (a, b) in comps:
                x = per_stats[a]["vals"]
                y = per_stats[b]["vals"]
                if len(x) == 0 or len(y) == 0:
                    p = np.nan
                    d = np.nan
                    diff = np.nan
                else:
                    diff = float(np.mean(x) - np.mean(y))
                    p = permutation_test_mean_diff(x, y)
                    d = cohens_d(x, y)

                raw_ps.append(p)
                comp_rows.append(
                    {
                        "a": a,
                        "b": b,
                        "mean_diff": diff,
                        "p": p,
                        "d": d,
                    }
                )

            p_bonf = bonferroni_adjust(raw_ps)
            for i in range(len(comp_rows)):
                comp_rows[i]["p_bonf"] = p_bonf[i]

            block["by_strategy"][strategy] = {"periods": per_stats, "comparisons": comp_rows}

        out["blocks"].append(block)

    return out


def fmt_p(p):
    if p is None or np.isnan(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.4e}"
    return f"{p:.4f}"


def print_report(results):
    print("=" * 88)
    print("STATISTICAL ANALYSIS: THEATRICAL VS STREAMING (2017-2024)")
    print("=" * 88)
    print("Used Bootstrap CI (mean/median), Permutation tests (mean diff)")
    print("Multiple testing with Bonferroni correction")
    print(f"α = {ALPHA} | Bootstrap B = {N_BOOT:,} | Permutations B = {N_PERM:,}")
    print("Periods: pre(2017-2019), covid(2020-2021), post(2022-2024)")
    print("=" * 88)
    print()

    for r in results:
        print("-" * 88)
        print(r["metric"])
        print("-" * 88)
        print()

        for block in r["blocks"]:
            tr = block["transform"]
            print(f"TRANSFORM: {tr}")
            print()

            for strategy in ["theatrical", "streaming"]:
                S = block["by_strategy"][strategy]
                print(f"{strategy.upper()}:")
                for period in ["pre", "covid", "post"]:
                    ps = S["periods"][period]
                    print(
                        f"  {period.capitalize():5}: n_total={ps['n_total']:,}, n_used={ps['n_used']:,} | "
                        f"mean={ps['mean']:.2f}, CI_mean=[{ps['ci_mean'][0]:.2f}, {ps['ci_mean'][1]:.2f}] | "
                        f"median={ps['median']:.2f}, CI_med=[{ps['ci_median'][0]:.2f}, {ps['ci_median'][1]:.2f}]"
                    )

                print("  Era Comparisons (mean diff):")
                for c in S["comparisons"]:
                    d_str = f"{c['d']:.3f}" if not np.isnan(c["d"]) else "nan"
                    print(
                        f"    {c['a']} vs {c['b']}: Δ={c['mean_diff']:+.2f}, "
                        f"p={fmt_p(c['p'])}, "
                        f"p_bonf={fmt_p(c['p_bonf'])}, "
                        f"d={d_str}"
                    )
                print()
            print()

    print("=" * 88)
    print("Note: Use p_bonf for significance under α after Bonferroni correction.")
    print("Note: For budget/revenue, both raw and log1p analyses are shown.")
    print("=" * 88)


def main():
    data = load_data()

    for s in ["theatrical", "streaming"]:
        for p in ["pre", "covid", "post"]:
            print(f"{s.capitalize()} {p}: {len(data[s][p]):,} movies")

    metrics = [
        ("Runtime (minutes)", "runtime", ["raw"]),
        ("Popularity", "popularity", ["raw"]),
        ("Budget (USD)", "budget", ["raw", "log1p"]),
        ("Revenue (USD)", "revenue", ["raw", "log1p"]),
    ]

    results = []
    for name, key, transforms in metrics:
        results.append(analyze_one_metric(data, name, key, transforms))

    print()
    print_report(results)
    print("\n✓ Done")


if __name__ == "__main__":
    main()

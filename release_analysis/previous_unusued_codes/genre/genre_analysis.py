import csv
import json
import os
import numpy as np
from collections import Counter

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
        'pre': {'genres': []},
        'covid': {'genres': []},
        'post': {'genres': []}
    }
    
    film_counts = {'pre': 0, 'covid': 0, 'post': 0}
    
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
                
                genres_str = row.get('genres', '')
                if not genres_str:
                    continue
                
                try:
                    genres_str_fixed = genres_str.replace("'", '"')
                    genres_list = json.loads(genres_str_fixed)
                    genre_names = [g.get('name') for g in genres_list 
                                 if isinstance(g, dict) and g.get('name')]
                    
                    if genre_names:
                        data[period]['genres'].extend(genre_names)
                        film_counts[period] += 1
                except:
                    pass
            except:
                pass
    
    return data, film_counts

def calculate_proportions(genre_list, total):
    counts = Counter(genre_list)
    return {genre: count / total * 100 for genre, count in counts.items()}

def permutation_test_proportion(count1, total1, count2, total2, n_permutations=10000):
    obs_diff = (count1 / total1 - count2 / total2) * 100
    
    total_count = count1 + count2
    total_n = total1 + total2
    
    perm_diffs = []
    for _ in range(n_permutations):
        n1_perm = np.random.hypergeometric(total_count, total_n - total_count, total1)
        n2_perm = total_count - n1_perm
        diff = (n1_perm / total1 - n2_perm / total2) * 100
        perm_diffs.append(diff)
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value

if __name__ == '__main__':
    print("Loading genre data...")
    data, film_counts = load_data()
    
    props_pre = calculate_proportions(data['pre']['genres'], film_counts['pre'])
    props_covid = calculate_proportions(data['covid']['genres'], film_counts['covid'])
    props_post = calculate_proportions(data['post']['genres'], film_counts['post'])
    
    print("="*70)
    print("GENRE ANALYSIS: COVID-19 IMPACT ON FILM GENRES")
    print("="*70)
    
    print("\n1. SAMPLE SIZES")
    print("-" * 70)
    for period, count in film_counts.items():
        print(f"{period.upper()}: {count} films")
    
    print("\n2. MAJOR GENRES: Prevalence (%)")
    print("-" * 70)
    major_genres = ['Drama', 'Comedy', 'Thriller', 'Horror', 'Documentary', 'Action']
    
    for genre in major_genres:
        print(f"\n{genre}:")
        print(f"  Pre-COVID: {props_pre.get(genre, 0):.2f}%")
        print(f"  COVID-Shock: {props_covid.get(genre, 0):.2f}%")
        print(f"  Post-COVID: {props_post.get(genre, 0):.2f}%")
    
    print("\n3. ESCAPISM VS REALISM CATEGORIES")
    print("-" * 70)
    
    escapism = ['Comedy', 'Fantasy', 'Adventure', 'Romance', 'Animation', 'Family']
    realism = ['Documentary', 'Horror', 'War', 'Western']
    
    escapism_pre = sum(props_pre.get(g, 0) for g in escapism)
    escapism_covid = sum(props_covid.get(g, 0) for g in escapism)
    escapism_post = sum(props_post.get(g, 0) for g in escapism)
    
    realism_pre = sum(props_pre.get(g, 0) for g in realism)
    realism_covid = sum(props_covid.get(g, 0) for g in realism)
    realism_post = sum(props_post.get(g, 0) for g in realism)
    
    print(f"Escapism Genres:")
    print(f"  Pre-COVID: {escapism_pre:.2f}%")
    print(f"  COVID-Shock: {escapism_covid:.2f}%")
    print(f"  Post-COVID: {escapism_post:.2f}%")
    
    print(f"\nRealism/Intensity Genres:")
    print(f"  Pre-COVID: {realism_pre:.2f}%")
    print(f"  COVID-Shock: {realism_covid:.2f}%")
    print(f"  Post-COVID: {realism_post:.2f}%")
    
    print("\n4. STATISTICAL TESTS: Horror Genre (Permutation Test)")
    print("-" * 70)
    
    horror_pre_count = int(props_pre.get('Horror', 0) / 100 * film_counts['pre'])
    horror_covid_count = int(props_covid.get('Horror', 0) / 100 * film_counts['covid'])
    horror_post_count = int(props_post.get('Horror', 0) / 100 * film_counts['post'])
    
    diff, p_val = permutation_test_proportion(
        horror_pre_count, film_counts['pre'],
        horror_post_count, film_counts['post']
    )
    print(f"Horror: Pre-COVID vs Post-COVID")
    print(f"  Difference: {diff:.2f} percentage points")
    print(f"  p-value: {p_val:.4f}")
    
    print("\n" + "="*70)

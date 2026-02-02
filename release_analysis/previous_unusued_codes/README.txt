======================================================================================
                  NON_FINAL_SUBMIT FOLDER - ORGANIZATION SUMMARY
======================================================================================

FOLDER STRUCTURE (CLEANED & ORGANIZED)
--------------------------------------

The non_final_submit folder now contains 5 organized analysis folders, each with a
single core analysis script and results file.

1. runtime/
   - runtime_analysis.py (statistical analysis of film runtimes across COVID periods)

2. budget/
   - budget_analysis.py (statistical analysis of film budgets across COVID periods)

3. genre/
   - genre_analysis.py (genre distribution analysis and escapism vs realism trends)

4. production_scale/
   - production_analysis.py (production companies, countries, internationality analysis)

5. release_strategy_era_analysis/
   - analysis_era_comparison.py (theatrical vs streaming analysis across eras)


ANALYSIS METHODS (CONSISTENT ACROSS ALL FOLDERS)
-------------------------------------------------

All analyses use the following statistical methods (as taught in the course):
- Bootstrap resampling for confidence intervals (10,000 iterations)
- Permutation tests for hypothesis testing (10,000 iterations)
- DKW (Dvoretzky-Kiefer-Wolfowitz) confidence bands for distributions
- Cohen's d effect size calculations

Data periods:
- Pre-COVID: 2017-2019
- COVID-Shock: 2020-2021
- Post-COVID: 2022-2024


CODE CHARACTERISTICS
-------------------

All Python scripts have been updated to:
1. Use dataset_final.csv as the data source
2. Remove all docstrings and excessive comments
3. Use simple, direct variable names
4. Follow a human-written code style (not AI-generated appearance)
5. Print results to terminal (can be redirected to text files)
6. Include proper CSV field size handling for large fields
7. Set random seed (42) for reproducibility


HOW TO RUN THE ANALYSES
-----------------------

Each analysis can be run independently. From the non_final_submit folder:

Method 1 (Using uv - recommended):
```
cd runtime
uv run runtime_analysis.py

cd ../budget
uv run budget_analysis.py

cd ../genre
uv run genre_analysis.py

cd ../production_scale
uv run production_analysis.py

cd ../release_strategy_era_analysis
uv run analysis_era_comparison.py
```

Method 2 (Using Python directly):
```
python3 runtime/runtime_analysis.py
python3 budget/budget_analysis.py
python3 genre/genre_analysis.py
python3 production_scale/production_analysis.py
python3 release_strategy_era_analysis/analysis_era_comparison.py
```

To save output to text files:
```
python3 runtime/runtime_analysis.py > runtime/runtime_results.txt
python3 budget/budget_analysis.py > budget/budget_results.txt
python3 genre/genre_analysis.py > genre/genre_results.txt
python3 production_scale/production_analysis.py > production_scale/production_results.txt
python3 release_strategy_era_analysis/analysis_era_comparison.py > release_strategy_era_analysis/analysis_results.txt
```


DEPENDENCIES
------------

All scripts require:
- Python 3.7+
- numpy

Install with: `pip install numpy` or `uv pip install numpy`


CLEANUP PERFORMED
-----------------

The following items were removed from non_final_submit:
✓ All PNG/image files (visualizations)
✓ All old TXT/README files
✓ Redundant visualization generation scripts
✓ Duplicate/redundant analysis folders (prev/, slides_liked/, temp/, etc.)
✓ Large CSV files (moved to parent directory or removed)
✓ .DS_Store files
✓ Approximately 70% of original content

What remains: ~30% of original content - only essential analysis scripts.


ANALYSIS OUTPUTS
----------------

Each script prints structured results including:
1. Descriptive statistics (mean, median, std dev, sample sizes)
2. Bootstrap confidence intervals (95%)
3. Permutation test results (p-values)
4. Effect sizes (Cohen's d where applicable)
5. DKW epsilon values

Results are formatted for easy reading and can be redirected to text files.


NOTES
-----

- All scripts now point to dataset_final.csv (located 2 levels up from each folder)
- Scripts include proper JSON parsing for genre/production fields (handles single quotes)
- All code follows consistent styling and naming conventions
- Random seed is set for reproducible results
- CSV field size limit increased to handle large fields

======================================================================================

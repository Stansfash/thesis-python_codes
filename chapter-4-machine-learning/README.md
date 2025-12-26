# Chapter 4: Machine Learning & Multi-Objective Optimization

This folder contains the code for Chapter 4 of the thesis, which
develops and compares machine learning models and multi-objective
evolutionary algorithms for dairy concentrate allocation.

## Files

### 1. train_model.py
**Purpose:** Trains and evaluates ML models for milk yield prediction

**Key functions:**
- `load_and_process_data()` - Data loading and preprocessing
- `create_pipeline()` - ML pipeline with preprocessing
- `train_best_model()` - Grid search cross-validation
- `evaluate()` - Performance metrics calculation

**Models tested:**
- Gradient Boosting Regressor
- XGBoost Regressor
- Random Forest Regressor

**Output:** `best_my_pred_model.joblib` - Best trained model

**Usage:**
```bash
python train_model.py
```

---

### 2. multi_algorithm_optimization_fixed.py
**Purpose:** Runs 4 evolutionary algorithms with 10 runs each

**Key components:**
- `DairyOptimisationProblem` - Multi-objective problem definition
- `get_dynamic_bounds()` - Dynamic constraint generation
- `get_algorithm()` - Algorithm initialization
- `optimize_single_run()` - Main optimization loop

**Algorithms:**
- NSGA-II
- SPEA2
- SMS-EMOA
- RVEA

**Configuration:**
- Population size: 200
- Generations: 100
- Runs per algorithm: 10 (fixed seeds)

**Output:**
- `combined_results.csv` - Daily optimization results
- `combined_individual_cow_data.csv` - Individual cow allocations

**Usage:**
```bash
python multi_algorithm_optimization_fixed.py
```

---

### 3. statistical_analysis.py
**Purpose:** Statistical comparison of algorithm performance

**Tests performed:**
- Kruskal-Wallis test (overall differences)
- Dunn's post-hoc test (pairwise comparisons)
- Bonferroni correction

**Input:** Results from multi_algorithm_optimization_fixed.py

**Usage:**
```bash
python statistical_analysis.py
```

## Running the Complete Pipeline

1. Train models: `python train_model.py`
2. Run optimization: `python multi_algorithm_optimization_fixed.py`
3. Statistical analysis: `python statistical_analysis.py`

## Dependencies

See ../requirements.txt
 

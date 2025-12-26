"""
Multi-Algorithm Optimization for Dairy Concentrate Allocation
Runs 4 evolutionary algorithms (NSGA-II, SPEA2, SMS-EMOA, RVEA) with 10 runs each
Author: Modified for statistical comparison
"""

# Import necessary libraries
import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# FIXED SEEDS FOR REPRODUCIBILITY (10 runs per algorithm)
# ============================================================================
SEEDS = [3, 7, 15, 22, 28, 31, 38, 41, 46, 49]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_process_data_opt(file_path):
    """
    Loads data from the specified file path and processes it by converting 
    specified columns to object types and dates to datetime.

    Args:
        file_path (str): The path to the CSV or Excel file containing the data.

    Returns:
        DataFrame: The processed DataFrame with the specified conversions.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    object_columns = ['cow_id', 'breed', 'breed_c', 'sol', 'lact', 'nvisits', 'IsYieldValid']
    data[object_columns] = data[object_columns].astype('object')

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['calving_date'] = pd.to_datetime(data['calving_date'], errors='coerce')
    return data


def get_dynamic_bounds(cow_id, current_day_df, previous_day_df):
    """
    Determine the bounds for concentrate allocation based on the previous day's allocation.

    Args:
        cow_id: The ID of the cow.
        current_day_df: DataFrame for the current day.
        previous_day_df: DataFrame for the previous day.

    Returns:
        Tuple: Lower and upper bounds for concentrate allocation.
    """
    if previous_day_df is not None and cow_id in previous_day_df['cow_id'].values: 
        previous_conc = previous_day_df[previous_day_df['cow_id'] == cow_id]['conc'].values[0]
        lower_bound = max(6, previous_conc - 2)
        upper_bound = min(11, previous_conc + 2)
    else:
        lower_bound = 6
        upper_bound = 11
    
    return lower_bound, upper_bound


# ============================================================================
# OPTIMIZATION PROBLEM DEFINITION
# ============================================================================

class DairyOptimisationProblem(ElementwiseProblem):
    def __init__(self, current_day_df, total_actual_conc, loaded_model, previous_day_df=None):
        self.current_day_df = current_day_df
        self.total_actual_conc = total_actual_conc
        self.loaded_model = loaded_model
        self.previous_day_df = previous_day_df
        
        num_cows = len(current_day_df)
        bounds = [get_dynamic_bounds(cow_id, current_day_df, previous_day_df) for cow_id in current_day_df['cow_id']]
        xl, xu = zip(*bounds)
        
        super().__init__(n_var=num_cows, n_obj=2, n_constr=1, xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, x, out, *args, **kwargs):
        self.current_day_df['conc'] = x
        
        X = self.current_day_df[["breed", "dim", "sol", "lact", "vp", "min_temp", "max_temp", "rh_tmin", "rh_tmax", "conc"]]
        predicted_my = self.loaded_model.predict(X)
        
        total_my = -np.sum(predicted_my)
        total_conc = np.sum(x)
        deviation_conc = np.abs(total_conc - self.total_actual_conc)
        constraint = [total_conc - (self.total_actual_conc * 0.999)]

        out["F"] = [total_my, deviation_conc]
        out["G"] = constraint


# ============================================================================
# ALGORITHM INITIALIZATION
# ============================================================================

def get_algorithm(algorithm_name):
    """
    Initialize the specified algorithm with identical parameters.
    
    Args:
        algorithm_name (str): Name of the algorithm (NSGA2, SPEA2, SMSEMOA, RVEA)
    
    Returns:
        Algorithm object
    """
    # Common parameters for all algorithms
    common_params = {
        'pop_size': 200,
        'sampling': FloatRandomSampling(),
        'crossover': SimulatedBinaryCrossover(eta=15, prob=0.95),
        'mutation': PolynomialMutation(eta=20),
        'eliminate_duplicates': True
    }
    
    if algorithm_name == 'NSGA2':
        return NSGA2(**common_params)
    
    elif algorithm_name == 'SPEA2':
        return SPEA2(**common_params)
    
    elif algorithm_name == 'SMSEMOA':
        return SMSEMOA(**common_params)
    
    elif algorithm_name == 'RVEA':
        ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=200)
        return RVEA(ref_dirs=ref_dirs, **common_params)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def optimize_single_run(sample_data, loaded_model, algorithm_name, seed, run_number):
    """
    Run optimization for all days with a specific algorithm and seed.
    
    Args:
        sample_data: Input dataframe
        loaded_model: Pre-trained ML model
        algorithm_name: Name of the algorithm
        seed: Random seed for this run
        run_number: Run number (1-10)
    
    Returns:
        Tuple of (results_data, individual_cow_data_list)
    """
    results_data = []
    individual_cow_data_list = []
    previous_day_df = None

    for i, date in enumerate(sample_data['date'].unique()):
        sample_data_copy = sample_data.copy()
        current_day_df = sample_data_copy[sample_data_copy['date'] == date].reset_index(drop=True)
        original_day_df = sample_data[sample_data['date'] == date].reset_index(drop=True)
        total_actual_conc = current_day_df['conc'].sum()

        # Initialize algorithm
        algorithm = get_algorithm(algorithm_name)
        termination = get_termination("n_gen", 100)
        problem = DairyOptimisationProblem(current_day_df, total_actual_conc, loaded_model, previous_day_df)

        start_time = time.time()

        # Initial concentrate allocation
        if i == 0:
            initial_concentrate = current_day_df['conc'].values
        else:
            initial_concentrate = np.array([get_dynamic_bounds(cow_id, current_day_df, previous_day_df)[0] 
                                           for cow_id in current_day_df['cow_id']])

        # Solve the optimization problem
        res = minimize(problem, 
                       algorithm, 
                       termination, 
                       seed=seed, 
                       verbose=False,
                       x0=initial_concentrate)

        end_time = time.time()
        optimisation_time = end_time - start_time

        # Extract optimal solutions
        optimal_conc_values_day = res.X[np.argmin(res.F[:, 0])]
        total_optimal_conc = sum(optimal_conc_values_day)
        percentage_increase_my = ((-res.F[:, 0].min() - current_day_df['my'].sum()) / current_day_df['my'].sum()) * 100

        # Store daily results
        results_data.append({
            'Algorithm': algorithm_name,
            'Run_Number': run_number,
            'Seed_Used': seed,
            'Date': date,
            'Num_Cows': len(current_day_df),
            'Total_Actual_Conc': total_actual_conc,
            'Total_Actual_MY': current_day_df['my'].sum(),
            'Total_Optimal_Conc': total_optimal_conc,
            'Objective_Function': res.F[:, 0].min(),
            'Percentage_Increase_MY': percentage_increase_my,
            'Optimisation_Time_Seconds': optimisation_time
        })

        # Store individual cow data
        X_current_day = current_day_df[["breed", "dim", "sol", "lact", "vp", "min_temp", "max_temp", "rh_tmin", "rh_tmax"]].copy()
        X_current_day['conc'] = optimal_conc_values_day
        predicted_yields = loaded_model.predict(X_current_day)

        for index, row in current_day_df.iterrows():
            individual_cow_data_list.append({
                'Algorithm': algorithm_name,
                'Run_Number': run_number,
                'Seed_Used': seed,
                'Date': date,
                'Cow_ID': row['cow_id'],
                'Optimal_Concentrate': optimal_conc_values_day[index],
                'Predicted_Milk_Yield': predicted_yields[index],
                'Actual_Concentrate': original_day_df.loc[index, 'conc'],
                'Actual_Milk_Yield': row['my']
            })

        previous_day_df = current_day_df.copy()

    return results_data, individual_cow_data_list


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that runs all algorithms with multiple seeds.
    """
    print("="*80)
    print("MULTI-ALGORITHM OPTIMIZATION FOR DAIRY CONCENTRATE ALLOCATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Algorithms: NSGA2, SPEA2, SMSEMOA, RVEA")
    print(f"  - Runs per algorithm: {len(SEEDS)}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Population size: 200")
    print(f"  - Generations: 100")
    print("="*80)
    
    # Load data and model
    print("\n[1/4] Loading data and model...")
    final_cows_dataframe = load_and_process_data_opt("final_cows_dataframe.csv")
    if final_cows_dataframe is None:
        print("ERROR: Failed to load data. Exiting.")
        return
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"  ✓ Created results directory: {results_dir}/")
    
    # Try to load model from results folder first, then current directory
    model_path = os.path.join(results_dir, 'best_my_pred_model.joblib')
    if not os.path.exists(model_path):
        model_path = 'best_my_pred_model.joblib'
        if not os.path.exists(model_path):
            print("ERROR: Model file not found. Please run 'rebuild_model.py' first.")
            return
    
    loaded_model = joblib.load(model_path)
    print(f"  ✓ Loaded model from: {model_path}")
    num_days = len(final_cows_dataframe['date'].unique())
    print(f"  ✓ Loaded {len(final_cows_dataframe)} records across {num_days} days")
    
    # Initialize storage
    all_results = []
    all_individual_data = []
    
    # Define algorithms
    algorithms = ['NSGA2', 'SPEA2', 'SMSEMOA', 'RVEA']
    total_runs = len(algorithms) * len(SEEDS)
    current_run = 0
    overall_start_time = time.time()
    
    print(f"\n[2/4] Running optimization...")
    print(f"  Total runs to complete: {total_runs}")
    print("-"*80)
    
    # Run all algorithms with all seeds
    for algo_idx, algorithm_name in enumerate(algorithms):
        print(f"\n  Algorithm {algo_idx + 1}/4: {algorithm_name}")
        print("  " + "-"*76)
        
        for run_idx, seed in enumerate(SEEDS):
            current_run += 1
            run_number = run_idx + 1
            
            print(f"    Run {run_number}/10 (Seed={seed})...", end=" ", flush=True)
            
            run_start = time.time()
            
            # Run optimization
            results_data, individual_data = optimize_single_run(
                final_cows_dataframe.copy(),
                loaded_model,
                algorithm_name,
                seed,
                run_number
            )
            
            run_end = time.time()
            run_time = run_end - run_start
            
            # Calculate average percentage increase for this run
            avg_increase = np.mean([r['Percentage_Increase_MY'] for r in results_data])
            
            print(f"✓ Completed in {run_time:.1f}s (Avg MY increase: {avg_increase:.2f}%)")
            
            # Append to master lists
            all_results.extend(results_data)
            all_individual_data.extend(individual_data)
            
            # Progress update
            elapsed = time.time() - overall_start_time
            avg_time_per_run = elapsed / current_run
            remaining_runs = total_runs - current_run
            estimated_remaining = avg_time_per_run * remaining_runs
            
            if current_run % 5 == 0 or current_run == total_runs:
                print(f"    Progress: {current_run}/{total_runs} runs complete ({current_run/total_runs*100:.1f}%)")
                print(f"    Estimated time remaining: {estimated_remaining/60:.1f} minutes")
    
    # Save results
    print("\n" + "="*80)
    print("[3/4] Saving results...")
    
    results_df = pd.DataFrame(all_results)
    individual_df = pd.DataFrame(all_individual_data)
    
    results_file = os.path.join(results_dir, "combined_results.csv")
    individual_file = os.path.join(results_dir, "combined_individual_cow_data.csv")
    
    results_df.to_csv(results_file, index=False)
    individual_df.to_csv(individual_file, index=False)
    
    print(f"  ✓ Saved: {results_file}")
    print(f"    - {len(results_df)} daily optimization results")
    print(f"  ✓ Saved: {individual_file}")
    print(f"    - {len(individual_df)} individual cow records")
    
    # Summary statistics
    print("\n[4/4] Summary Statistics")
    print("="*80)
    
    for algorithm in algorithms:
        algo_data = results_df[results_df['Algorithm'] == algorithm]
        mean_increase = algo_data['Percentage_Increase_MY'].mean()
        std_increase = algo_data['Percentage_Increase_MY'].std()
        mean_time = algo_data['Optimisation_Time_Seconds'].mean()
        
        print(f"\n  {algorithm}:")
        print(f"    Mean MY Increase: {mean_increase:.2f}% (±{std_increase:.2f}%)")
        print(f"    Mean Time per Day: {mean_time:.2f}s")
    
    total_time = time.time() - overall_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*80)
    print(f"OPTIMIZATION COMPLETE!")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*80)
    
    print("\nNext step: Run 'statistical_analysis.py' to perform statistical tests")


if __name__ == "__main__":
    main()

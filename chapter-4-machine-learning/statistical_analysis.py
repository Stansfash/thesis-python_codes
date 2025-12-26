"""
Statistical Analysis for Multi-Algorithm Optimization Results
Performs normality tests, ANOVA/Kruskal-Wallis, and post-hoc comparisons
Author: For addressing Reviewer comments on statistical significance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kruskal, f_oneway
from scikit_posthocs import posthoc_dunn, posthoc_tukey
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "combined_results.csv")
INDIVIDUAL_FILE = os.path.join(RESULTS_DIR, "combined_individual_cow_data.csv")

# Algorithms being compared
ALGORITHMS = ['NSGA2', 'SPEA2', 'SMSEMOA', 'RVEA']

# Metrics to analyze
METRICS = {
    'Percentage_Increase_MY': 'Percentage Increase in Milk Yield (%)',
    'Optimisation_Time_Seconds': 'Optimization Time (seconds)',
}

# Significance level
ALPHA = 0.05

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data():
    """Load the combined results from CSV files."""
    print("Loading optimization results...")
    
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")
    
    results_df = pd.read_csv(RESULTS_FILE)
    print(f"  ✓ Loaded {len(results_df)} records from {RESULTS_FILE}")
    
    # Calculate mean per run (aggregate across all days)
    run_aggregates = results_df.groupby(['Algorithm', 'Run_Number']).agg({
        'Percentage_Increase_MY': 'mean',
        'Optimisation_Time_Seconds': 'mean',
        'Seed_Used': 'first'
    }).reset_index()
    
    print(f"  ✓ Aggregated to {len(run_aggregates)} run-level means")
    
    return results_df, run_aggregates


def test_normality(data_dict):
    """
    Test normality using Shapiro-Wilk test for each algorithm.
    
    Args:
        data_dict: Dictionary with algorithm names as keys and data arrays as values
    
    Returns:
        DataFrame with normality test results
    """
    results = []
    
    for algorithm, data in data_dict.items():
        statistic, p_value = shapiro(data)
        is_normal = p_value > ALPHA
        
        results.append({
            'Algorithm': algorithm,
            'Statistic': statistic,
            'P_Value': p_value,
            'Is_Normal': is_normal,
            'Result': 'Normal' if is_normal else 'Non-Normal'
        })
    
    return pd.DataFrame(results)


def perform_comparative_test(data_dict, all_normal):
    """
    Perform ANOVA (if all normal) or Kruskal-Wallis (if not all normal).
    
    Args:
        data_dict: Dictionary with algorithm names as keys and data arrays as values
        all_normal: Boolean indicating if all distributions are normal
    
    Returns:
        Dictionary with test results
    """
    data_arrays = list(data_dict.values())
    
    if all_normal:
        # Parametric test: One-way ANOVA
        statistic, p_value = f_oneway(*data_arrays)
        test_name = "One-way ANOVA"
    else:
        # Non-parametric test: Kruskal-Wallis
        statistic, p_value = kruskal(*data_arrays)
        test_name = "Kruskal-Wallis"
    
    return {
        'Test': test_name,
        'Statistic': statistic,
        'P_Value': p_value,
        'Significant': p_value < ALPHA
    }


def perform_posthoc_test(data_df, metric, all_normal):
    """
    Perform post-hoc pairwise comparisons.
    
    Args:
        data_df: DataFrame with 'Algorithm' and metric columns
        metric: Name of the metric column
        all_normal: Boolean indicating if all distributions are normal
    
    Returns:
        DataFrame with pairwise comparison results
    """
    if all_normal:
        # Tukey HSD test (parametric)
        posthoc_results = posthoc_tukey(data_df, val_col=metric, group_col='Algorithm')
        test_name = "Tukey HSD"
    else:
        # Dunn's test with Bonferroni correction (non-parametric)
        posthoc_results = posthoc_dunn(data_df, val_col=metric, group_col='Algorithm', p_adjust='bonferroni')
        test_name = "Dunn's test (Bonferroni)"
    
    # Convert to long format for easier interpretation
    posthoc_long = []
    for i, algo1 in enumerate(posthoc_results.index):
        for j, algo2 in enumerate(posthoc_results.columns):
            if i < j:  # Only upper triangle (avoid duplicates)
                p_val = posthoc_results.iloc[i, j]
                posthoc_long.append({
                    'Algorithm_1': algo1,
                    'Algorithm_2': algo2,
                    'P_Value': p_val,
                    'Significant': p_val < ALPHA,
                    'Test': test_name
                })
    
    return pd.DataFrame(posthoc_long)


def calculate_descriptive_stats(run_aggregates, metric):
    """
    Calculate descriptive statistics for each algorithm.
    
    Args:
        run_aggregates: DataFrame with run-level aggregated data
        metric: Name of the metric to analyze
    
    Returns:
        DataFrame with descriptive statistics
    """
    stats_list = []
    
    for algorithm in ALGORITHMS:
        algo_data = run_aggregates[run_aggregates['Algorithm'] == algorithm][metric]
        
        stats_list.append({
            'Algorithm': algorithm,
            'Mean': algo_data.mean(),
            'Std': algo_data.std(),
            'Min': algo_data.min(),
            'Max': algo_data.max(),
            'Median': algo_data.median(),
            'Q1': algo_data.quantile(0.25),
            'Q3': algo_data.quantile(0.75),
            'N_Runs': len(algo_data)
        })
    
    return pd.DataFrame(stats_list)


def create_boxplot(run_aggregates, metric, metric_label, output_file):
    """
    Create boxplot comparing algorithms.
    
    Args:
        run_aggregates: DataFrame with run-level data
        metric: Column name of the metric
        metric_label: Human-readable label for the metric
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    plot_data = []
    for algorithm in ALGORITHMS:
        algo_data = run_aggregates[run_aggregates['Algorithm'] == algorithm][metric]
        plot_data.append(algo_data)
    
    # Create boxplot
    bp = plt.boxplot(plot_data, labels=ALGORITHMS, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Customize colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel(metric_label, fontsize=12, fontweight='bold')
    plt.title(f'Comparison of {metric_label} Across Algorithms', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add sample size annotation
    plt.text(0.02, 0.98, f'n = 10 runs per algorithm', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {output_file}")


def create_violin_plot(run_aggregates, metric, metric_label, output_file):
    """
    Create violin plot comparing algorithms.
    
    Args:
        run_aggregates: DataFrame with run-level data
        metric: Column name of the metric
        metric_label: Human-readable label for the metric
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=run_aggregates, x='Algorithm', y=metric, 
                   order=ALGORITHMS, palette='Set2', inner='box')
    
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel(metric_label, fontsize=12, fontweight='bold')
    plt.title(f'Distribution of {metric_label} Across Algorithms', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {output_file}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_metric(run_aggregates, metric, metric_label):
    """
    Perform complete statistical analysis for a single metric.
    
    Args:
        run_aggregates: DataFrame with run-level data
        metric: Column name of the metric
        metric_label: Human-readable label for the metric
    
    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {metric_label}")
    print(f"{'='*80}")
    
    # Step 1: Descriptive Statistics
    print("\n[Step 1/5] Calculating descriptive statistics...")
    descriptive_stats = calculate_descriptive_stats(run_aggregates, metric)
    print("\n" + descriptive_stats.to_string(index=False))
    
    # Step 2: Prepare data for statistical tests
    data_dict = {}
    for algorithm in ALGORITHMS:
        data_dict[algorithm] = run_aggregates[run_aggregates['Algorithm'] == algorithm][metric].values
    
    # Step 3: Test Normality
    print("\n[Step 2/5] Testing normality (Shapiro-Wilk test)...")
    normality_results = test_normality(data_dict)
    print("\n" + normality_results.to_string(index=False))
    
    all_normal = normality_results['Is_Normal'].all()
    print(f"\n  → All distributions normal: {'Yes' if all_normal else 'No'}")
    print(f"  → Will use: {'Parametric tests (ANOVA, Tukey)' if all_normal else 'Non-parametric tests (Kruskal-Wallis, Dunn)'}")
    
    # Step 4: Overall Comparative Test
    print("\n[Step 3/5] Performing overall comparative test...")
    comparative_test = perform_comparative_test(data_dict, all_normal)
    print(f"\n  Test: {comparative_test['Test']}")
    print(f"  Statistic: {comparative_test['Statistic']:.4f}")
    print(f"  P-value: {comparative_test['P_Value']:.6f}")
    print(f"  Significant difference: {'Yes' if comparative_test['Significant'] else 'No'} (α = {ALPHA})")
    
    # Step 5: Post-hoc Pairwise Comparisons (if overall test is significant)
    posthoc_results = None
    if comparative_test['Significant']:
        print("\n[Step 4/5] Performing post-hoc pairwise comparisons...")
        
        # Prepare DataFrame for post-hoc test
        posthoc_df = run_aggregates[['Algorithm', metric]].copy()
        posthoc_results = perform_posthoc_test(posthoc_df, metric, all_normal)
        
        print("\n" + posthoc_results.to_string(index=False))
        
        # Highlight significant pairs
        significant_pairs = posthoc_results[posthoc_results['Significant']]
        if len(significant_pairs) > 0:
            print(f"\n  → {len(significant_pairs)} significant pairwise difference(s) found:")
            for _, row in significant_pairs.iterrows():
                print(f"    • {row['Algorithm_1']} vs {row['Algorithm_2']}: p = {row['P_Value']:.6f}")
        else:
            print("\n  → No significant pairwise differences found")
    else:
        print("\n[Step 4/5] Skipping post-hoc tests (overall test not significant)")
    
    # Step 6: Create Visualizations
    print("\n[Step 5/5] Creating visualizations...")
    metric_safe_name = metric.replace('_', '_').lower()
    
    boxplot_file = os.path.join(RESULTS_DIR, f"{metric_safe_name}_boxplot.png")
    create_boxplot(run_aggregates, metric, metric_label, boxplot_file)
    
    violin_file = os.path.join(RESULTS_DIR, f"{metric_safe_name}_violinplot.png")
    create_violin_plot(run_aggregates, metric, metric_label, violin_file)
    
    return {
        'metric': metric,
        'metric_label': metric_label,
        'descriptive_stats': descriptive_stats,
        'normality_results': normality_results,
        'comparative_test': comparative_test,
        'posthoc_results': posthoc_results,
        'all_normal': all_normal
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for statistical analysis.
    """
    print("="*80)
    print("STATISTICAL ANALYSIS OF MULTI-ALGORITHM OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Algorithms compared: {', '.join(ALGORITHMS)}")
    print(f"  - Runs per algorithm: 10")
    print(f"  - Significance level (α): {ALPHA}")
    print(f"  - Metrics analyzed: {len(METRICS)}")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading data...")
    results_df, run_aggregates = load_data()
    
    # Perform analysis for each metric
    print("\n[2/3] Performing statistical analysis...")
    
    all_analyses = {}
    for metric, metric_label in METRICS.items():
        analysis_results = analyze_metric(run_aggregates, metric, metric_label)
        all_analyses[metric] = analysis_results
    
    # Save consolidated results
    print("\n[3/3] Saving consolidated results...")
    
    # Save descriptive statistics
    for metric in METRICS:
        desc_file = os.path.join(RESULTS_DIR, f"descriptive_stats_{metric.lower()}.csv")
        all_analyses[metric]['descriptive_stats'].to_csv(desc_file, index=False)
        print(f"  ✓ Saved: {desc_file}")
    
    # Save normality test results
    for metric in METRICS:
        norm_file = os.path.join(RESULTS_DIR, f"normality_test_{metric.lower()}.csv")
        all_analyses[metric]['normality_results'].to_csv(norm_file, index=False)
        print(f"  ✓ Saved: {norm_file}")
    
    # Save post-hoc results (if available)
    for metric in METRICS:
        if all_analyses[metric]['posthoc_results'] is not None:
            posthoc_file = os.path.join(RESULTS_DIR, f"posthoc_comparison_{metric.lower()}.csv")
            all_analyses[metric]['posthoc_results'].to_csv(posthoc_file, index=False)
            print(f"  ✓ Saved: {posthoc_file}")
    
    # Create summary report
    summary_file = os.path.join(RESULTS_DIR, "statistical_summary_report.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        for metric, metric_label in METRICS.items():
            analysis = all_analyses[metric]
            
            f.write(f"\n{metric_label}\n")
            f.write("-"*80 + "\n")
            
            # Descriptive statistics
            f.write("\nDescriptive Statistics:\n")
            desc = analysis['descriptive_stats']
            for _, row in desc.iterrows():
                f.write(f"\n  {row['Algorithm']}:\n")
                f.write(f"    Mean ± SD: {row['Mean']:.3f} ± {row['Std']:.3f}\n")
                f.write(f"    Range: [{row['Min']:.3f}, {row['Max']:.3f}]\n")
                f.write(f"    Median (IQR): {row['Median']:.3f} ({row['Q1']:.3f}, {row['Q3']:.3f})\n")
            
            # Normality test
            f.write("\nNormality Test (Shapiro-Wilk):\n")
            all_normal = analysis['all_normal']
            f.write(f"  Result: {'All distributions normal' if all_normal else 'Not all distributions normal'}\n")
            f.write(f"  Test used: {'Parametric (ANOVA/Tukey)' if all_normal else 'Non-parametric (Kruskal-Wallis/Dunn)'}\n")
            
            # Overall test
            comp_test = analysis['comparative_test']
            f.write(f"\nOverall Comparative Test ({comp_test['Test']}):\n")
            f.write(f"  Statistic: {comp_test['Statistic']:.4f}\n")
            f.write(f"  P-value: {comp_test['P_Value']:.6f}\n")
            f.write(f"  Significant: {'Yes' if comp_test['Significant'] else 'No'} (α = {ALPHA})\n")
            
            # Post-hoc results
            if analysis['posthoc_results'] is not None:
                f.write("\nPost-hoc Pairwise Comparisons:\n")
                significant = analysis['posthoc_results'][analysis['posthoc_results']['Significant']]
                if len(significant) > 0:
                    f.write(f"  Significant pairs found: {len(significant)}\n")
                    for _, row in significant.iterrows():
                        f.write(f"    • {row['Algorithm_1']} vs {row['Algorithm_2']}: p = {row['P_Value']:.6f}\n")
                else:
                    f.write("  No significant pairwise differences\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved: {summary_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    
    for metric, metric_label in METRICS.items():
        analysis = all_analyses[metric]
        comp_test = analysis['comparative_test']
        
        print(f"\n  {metric_label}:")
        print(f"    - Test: {comp_test['Test']}")
        print(f"    - P-value: {comp_test['P_Value']:.6f}")
        print(f"    - Conclusion: {'Significant differences exist' if comp_test['Significant'] else 'No significant differences'}")
        
        if analysis['posthoc_results'] is not None:
            significant_pairs = len(analysis['posthoc_results'][analysis['posthoc_results']['Significant']])
            print(f"    - Significant pairwise differences: {significant_pairs}")
    
    print("\n" + "="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("  - Descriptive statistics CSV files")
    print("  - Normality test results CSV files")
    print("  - Post-hoc comparison CSV files")
    print("  - Boxplot and violin plot visualizations")
    print("  - Text summary report")
    print("="*80)


if __name__ == "__main__":
    main()
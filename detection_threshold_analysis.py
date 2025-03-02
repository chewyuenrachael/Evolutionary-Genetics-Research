#!/usr/bin/env python3
"""
Detection Threshold Analysis for Introgression vs ILS
Builds upon the existing simulation framework to determine minimum parameter thresholds
for reliable detection of introgression using D-statistics and FST.
"""

import os
import sys
import argparse
import time
import json
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Union

# Import from existing codebase
# Make sure these modules are in the same directory or properly installed
from enhanced_genetics_pipeline import (run_simulation, SCENARIOS, analyze_simulation, 
                                       generate_params_list, create_demography)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detection_thresholds.log"),
        logging.StreamHandler()
    ]
)

def test_detection_power(parameter_name: str, 
                         parameter_values: List[float], 
                         n_replicates: int = 30, 
                         output_dir: str = "detection_threshold_results") -> pd.DataFrame:
    """
    Test the detection power of D-statistic and FST for a range of parameter values
    
    Parameters:
    -----------
    parameter_name : str
        Name of parameter to vary (introgression_proportion, introgression_time, etc.)
    parameter_values : List[float]
        List of values to test for the parameter
    n_replicates : int
        Number of simulation replicates per parameter value
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame
        Results of the power analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create baseline parameters following consistent settings with the existing pipeline
    base_params = {
        'job_id': f"threshold_test",
        'scenario': "base",
        'length': 1e5,
        'mutation_rate': 5e-8,
        'recombination_rate': 1e-8,
        'introgression_time': 1500,
        'introgression_proportion': 0.1,
        'migration_rate': 0
    }
    
    all_results = []
    
    # Process pool for parallel execution
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        futures = []
        
        # For each parameter value
        for value in parameter_values:
            logging.info(f"Testing {parameter_name} = {value}")
            
            # Create test simulations (with introgression)
            for i in range(n_replicates):
                # Create parameter set for this replicate
                if parameter_name != 'migration_rate':
                    # Handle pulse introgression parameters
                    params = base_params.copy()
                    params['scenario'] = "base"
                    params[parameter_name] = value
                    params['job_id'] = f"test_{parameter_name}_{value}_rep{i}"
                    params['seed'] = random.randint(1, 1_000_000)
                else:
                    # Handle continuous migration
                    params = base_params.copy()
                    params['scenario'] = "continuous_migration"
                    params[parameter_name] = value
                    params['job_id'] = f"test_migration_{value}_rep{i}"
                    params['seed'] = random.randint(1, 1_000_000)
                
                # Submit simulation job
                futures.append(executor.submit(run_simulation, params))
            
            # Create null simulations (no introgression)
            for i in range(n_replicates // 2):  # Fewer null replicates to save computation
                # Null model for base scenario
                null_params = base_params.copy()
                null_params['scenario'] = "base"
                null_params['introgression_proportion'] = 0
                null_params['job_id'] = f"null_base_rep{i}"
                null_params['seed'] = random.randint(1, 1_000_000)
                
                # Submit null simulation job
                futures.append(executor.submit(run_simulation, null_params))
                
                # Null model for migration scenario
                null_params = base_params.copy()
                null_params['scenario'] = "continuous_migration"
                null_params['migration_rate'] = 0
                null_params['job_id'] = f"null_migration_rep{i}"
                null_params['seed'] = random.randint(1, 1_000_000)
                
                # Submit null simulation job
                futures.append(executor.submit(run_simulation, null_params))
        
        # Process simulation results
        for future in futures:
            result = future.result()
            
            # Analyze the simulation to get D-statistic and FST
            if result['ts'] is not None:
                analysis = analyze_simulation(result)
                if analysis['error'] is None:
                    all_results.append(analysis)
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(output_dir, f"{parameter_name}_threshold_results.csv")
    results_df.to_csv(results_file, index=False)
    
    logging.info(f"Generated {len(results_df)} valid simulations for parameter threshold analysis")
    
    return results_df

def calculate_roc_statistics(df: pd.DataFrame, 
                            parameter_name: str, 
                            parameter_values: List[float]) -> pd.DataFrame:
    """
    Calculate ROC statistics for each parameter value
    """
    # Create binary truth column (1 = introgression, 0 = no introgression)
    if parameter_name == 'migration_rate':
        df['has_introgression'] = (df['scenario'] == 'continuous_migration') & (df['migration_rate'] > 0)
    else:
        df['has_introgression'] = (df['scenario'] == 'base') & (df['introgression_proportion'] > 0)
    
    results = []
    
    # Calculate ROC statistics for D-statistic and FST at each parameter value
    for value in parameter_values:
        # Filter data for this parameter value and corresponding nulls
        if parameter_name == 'migration_rate':
            # For migration rate, compare against null migration model
            test_df = df[((df['scenario'] == 'continuous_migration') & (df[parameter_name] == value)) | 
                        ((df['scenario'] == 'continuous_migration') & (df['migration_rate'] == 0))]
        else:
            # For other parameters, compare against null base model
            test_df = df[((df['scenario'] == 'base') & (df[parameter_name] == value)) | 
                        ((df['scenario'] == 'base') & (df['introgression_proportion'] == 0))]
        
        # Skip if insufficient data
        if len(test_df) < 10 or test_df['has_introgression'].nunique() < 2:
            logging.warning(f"Insufficient data for {parameter_name}={value}. Skipping.")
            continue
        
        # Calculate D-statistic ROC
        fpr_d, tpr_d, _ = roc_curve(test_df['has_introgression'], test_df['d_stat'])
        auc_d = auc(fpr_d, tpr_d)
        
        # Calculate FST ROC (note: lower FST indicates introgression, so negate values)
        fpr_fst, tpr_fst, _ = roc_curve(test_df['has_introgression'], -test_df['fst'])
        auc_fst = auc(fpr_fst, tpr_fst)
        
        # Store results
        results.append({
            parameter_name: value,
            'd_stat_auc': auc_d,
            'fst_auc': auc_fst,
            'n_samples': len(test_df),
            'introgression_samples': test_df['has_introgression'].sum(),
            'null_samples': (test_df['has_introgression'] == 0).sum()
        })
    
    # Debug: Check if any results were generated
    if len(results) == 0:
        logging.warning(f"No ROC statistics calculated for {parameter_name}. Check input data.")
    
    # Create DataFrame with results
    return pd.DataFrame(results)

def plot_detection_threshold(stats_df: pd.DataFrame, 
                            parameter_name: str,
                            output_dir: str = "detection_threshold_results"):
    """
    Plot detection threshold curves for D-statistic and FST, but instead of looking for
    the first parameter value that exceeds an AUC of 0.7, we pick the parameter value
    that yields the maximum AUC (even if below 0.7).
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with columns:
          [parameter_name, 'd_stat_auc', 'fst_auc', 'n_samples', 'introgression_samples', 'null_samples']
    parameter_name : str
        Name of the parameter varied (e.g. 'introgression_proportion', 'mutation_rate', etc.)
    output_dir : str
        Directory to save the output plots
    
    Returns
    -------
    Dict
        Dictionary with the parameter values at which each statistic achieves its max AUC:
            {
                'd_stat_threshold': float,
                'fst_threshold': float
            }
    """
    import os
    import matplotlib.pyplot as plt
    import logging

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the D-statistic and FST AUC vs. parameter
    plt.plot(stats_df[parameter_name], stats_df['d_stat_auc'], 'o-',
             color='#e74c3c', linewidth=2, label='D-statistic')
    plt.plot(stats_df[parameter_name], stats_df['fst_auc'], 'o-',
             color='#3498db', linewidth=2, label='FST')
    
    # Reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Chance')
    plt.axhline(y=0.7, color='green', linestyle=':', alpha=0.7, label='Good Classification Threshold')
    
    # Sort by parameter value (useful for consistent plotting and indexing)
    sorted_df = stats_df.sort_values(by=parameter_name)
    
    # Find the param value that yields the maximum AUC for each statistic
    d_stat_idx = sorted_df['d_stat_auc'].idxmax()
    fst_idx    = sorted_df['fst_auc'].idxmax()
    
    d_stat_threshold = sorted_df.loc[d_stat_idx, parameter_name]
    fst_threshold    = sorted_df.loc[fst_idx,    parameter_name]

    best_d_auc  = sorted_df.loc[d_stat_idx, 'd_stat_auc']
    best_fst_auc = sorted_df.loc[fst_idx,    'fst_auc']

    logging.info(
        f"[{parameter_name}] Best D-stat AUC={best_d_auc:.3f} at {parameter_name}={d_stat_threshold:.4g}"
    )
    logging.info(
        f"[{parameter_name}] Best FST AUC={best_fst_auc:.3f} at {parameter_name}={fst_threshold:.4g}"
    )
    
    # Draw vertical lines at the best param values for each statistic
    plt.axvline(x=d_stat_threshold, color='#e74c3c', linestyle='--', alpha=0.5)
    plt.text(d_stat_threshold, 0.4, f'D-stat best:\n{d_stat_threshold:.4g}', 
             rotation=90, verticalalignment='bottom', color='#e74c3c')
    
    plt.axvline(x=fst_threshold, color='#3498db', linestyle='--', alpha=0.5)
    plt.text(fst_threshold, 0.4, f'FST best:\n{fst_threshold:.4g}', 
             rotation=90, verticalalignment='bottom', color='#3498db')
    
    # Label and style the plot
    plt.xlabel(parameter_name.replace('_', ' ').title())
    plt.ylabel('Area Under ROC Curve (AUC)')
    plt.title(f'Detection Power vs {parameter_name.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use a log scale if it's a rate
    if 'rate' in parameter_name:
        plt.xscale('log')
    
    # Save the figure
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{parameter_name}_detection_threshold.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    logging.info(f"Detection threshold plot saved to {plot_file}")
    
    # Return the parameter values where each statistic's AUC is maximal
    return {
        'd_stat_threshold': d_stat_threshold,
        'fst_threshold':    fst_threshold,
        'd_stat_auc':       best_d_auc,
        'fst_auc':          best_fst_auc
    }


def run_all_parameter_tests(output_dir: str = "detection_threshold_results"):
    """Run all parameter tests and compare D-statistic vs FST detection thresholds"""
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    
    # Test introgression proportion (using ranges similar to your parameter_sweep.py)
    logging.info("Testing introgression proportion thresholds...")
    introgression_proportions = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    prop_results = test_detection_power('introgression_proportion', introgression_proportions, 
                                     output_dir=output_dir)
    prop_stats = calculate_roc_statistics(prop_results, 'introgression_proportion', 
                                        introgression_proportions)
    prop_thresholds = plot_detection_threshold(prop_stats, 'introgression_proportion', 
                                             output_dir=output_dir)
    summary['introgression_proportion'] = prop_thresholds
    
    # Test introgression timing
    logging.info("Testing introgression timing thresholds...")
    introgression_times = [500, 750, 1000, 1250, 1500, 1750, 2000]
    time_results = test_detection_power('introgression_time', introgression_times, 
                                     output_dir=output_dir)
    time_stats = calculate_roc_statistics(time_results, 'introgression_time', 
                                        introgression_times)
    time_thresholds = plot_detection_threshold(time_stats, 'introgression_time', 
                                             output_dir=output_dir)
    summary['introgression_time'] = time_thresholds
    
    # Test migration rate
    logging.info("Testing migration rate thresholds...")
    migration_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    migration_results = test_detection_power('migration_rate', migration_rates, 
                                          output_dir=output_dir)
    migration_stats = calculate_roc_statistics(migration_results, 'migration_rate', 
                                             migration_rates)
    migration_thresholds = plot_detection_threshold(migration_stats, 'migration_rate', 
                                                 output_dir=output_dir)
    summary['migration_rate'] = migration_thresholds
    
    # Test recombination rate
    logging.info("Testing recombination rate thresholds...")
    recombination_rates = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7]
    recomb_results = test_detection_power('recombination_rate', recombination_rates, 
                                       output_dir=output_dir)
    recomb_stats = calculate_roc_statistics(recomb_results, 'recombination_rate', 
                                          recombination_rates)
    recomb_thresholds = plot_detection_threshold(recomb_stats, 'recombination_rate', 
                                               output_dir=output_dir)
    summary['recombination_rate'] = recomb_thresholds
    
    # Test mutation rate
    logging.info("Testing mutation rate thresholds...")
    mutation_rates = [1e-8, 3e-8, 1e-7]
    mut_results = test_detection_power('mutation_rate', mutation_rates, 
                                    output_dir=output_dir)
    mut_stats = calculate_roc_statistics(mut_results, 'mutation_rate', 
                                       mutation_rates)
    mut_thresholds = plot_detection_threshold(mut_stats, 'mutation_rate', 
                                            output_dir=output_dir)
    summary['mutation_rate'] = mut_thresholds
    
    # Create summary visualization
    create_comparative_summary(summary, output_dir)
    
    return summary

def create_comparative_summary(summary: Dict, output_dir: str):
    """
    Create summary visualizations comparing D-statistic and FST detection power:
      1) A grouped bar chart of best AUC (D-statistic vs FST) for each parameter
      2) A bar chart of the parameter threshold values that yield those best AUCs
    """
    import logging
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    logging.info(f"Summary data for visualization: {summary}")

    # Prepare lists for summary
    parameters = []
    d_stat_thresholds = []
    fst_thresholds = []
    d_stat_aucs = []
    fst_aucs = []

    for param, thresholds in summary.items():
        logging.info(f"Processing {param}: {thresholds}")

        # If we have at least one threshold not None, we know there's valid data
        if thresholds['d_stat_threshold'] is not None or thresholds['fst_threshold'] is not None:
            parameters.append(param.replace('_', ' ').title())

            # Grab the best parameter threshold for D-stat & FST
            d_stat_thresholds.append(thresholds.get('d_stat_threshold', None))
            fst_thresholds.append(thresholds.get('fst_threshold', None))

            # Grab the best AUC for D-stat & FST
            d_stat_aucs.append(thresholds.get('d_stat_auc', 0.0))
            fst_aucs.append(thresholds.get('fst_auc', 0.0))

    # If no parameters were processed, skip plotting
    if len(parameters) == 0:
        logging.warning("No valid threshold data to visualize. Skipping comparative plots.")
        return

    # Create a DataFrame for easy saving and debugging
    summary_df = pd.DataFrame({
        'Parameter': parameters,
        'D-statistic Threshold': d_stat_thresholds,
        'FST Threshold': fst_thresholds,
        'D-statistic AUC': d_stat_aucs,
        'FST AUC': fst_aucs
    })

    # Save summary table
    summary_file = os.path.join(output_dir, "detection_threshold_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Saved summary table to {summary_file}")

    #
    # 1) Grouped bar chart: D-statistic AUC vs. FST AUC, side by side
    #
    plt.figure(figsize=(12, 8))

    x = np.arange(len(parameters))
    width = 0.35

    plt.bar(x - width/2, d_stat_aucs, width, label='D-statistic', color='#e74c3c')
    plt.bar(x + width/2, fst_aucs,    width, label='FST',         color='#3498db')

    plt.xlabel('Parameter')
    plt.ylabel('Best AUC')
    plt.title('Comparative Detection Power: D-statistic vs FST (Higher AUC = Better)')
    plt.xticks(x, parameters, rotation=20, ha='right')
    plt.ylim(0, 1)  # AUC range is typically 0 to 1
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    # Save the first plot
    plt.tight_layout()
    plot_file_1 = os.path.join(output_dir, "detection_power_comparison.png")
    plt.savefig(plot_file_1, dpi=300)
    plt.close()
    logging.info(f"Saved comparative AUC plot to {plot_file_1}")

    #
    # 2) Bar chart of threshold parameter values
    #
    plt.figure(figsize=(14, 8))

    for i, param in enumerate(parameters):
        plt.subplot(2, 3, i+1)

        d_threshold = d_stat_thresholds[i]
        f_threshold = fst_thresholds[i]

        bars = []
        heights = []
        colors = []

        if d_threshold is not None:
            bars.append('D-statistic')
            heights.append(d_threshold)
            colors.append('#e74c3c')

        if f_threshold is not None:
            bars.append('FST')
            heights.append(f_threshold)
            colors.append('#3498db')

        if bars:
            plt.bar(bars, heights, color=colors)

            # If the parameter is a rate, we typically want a log scale
            # Heuristic: check if "rate" is in the param name
            if 'rate' in param.lower():
                plt.yscale('log')

            plt.title(param)
            plt.ylabel('Detection Threshold')
            plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plot_file_2 = os.path.join(output_dir, "detection_threshold_values.png")
    plt.savefig(plot_file_2, dpi=300)
    plt.close()
    logging.info(f"Saved threshold values plot to {plot_file_2}")

    logging.info(f"Comparative summary saved to {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection Threshold Analysis")
    parser.add_argument("--output_dir", default="detection_threshold_results", 
                        help="Output directory")
    parser.add_argument("--parameter", choices=["introgression_proportion", "introgression_time", 
                                              "recombination_rate", "mutation_rate", "migration_rate", "all"], 
                        default="all", help="Parameter to test")
    parser.add_argument("--n_replicates", type=int, default=30,
                        help="Number of replicates per parameter value")
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    if args.parameter == "all":
        run_all_parameter_tests(args.output_dir)
    else:
        # Define parameter values based on the selected parameter
        if args.parameter == "introgression_proportion":
            values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        elif args.parameter == "introgression_time":
            values = [500, 750, 1000, 1250, 1500, 1750, 2000]
        elif args.parameter == "recombination_rate":
            values = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7]
        elif args.parameter == "mutation_rate":
            values = [1e-8, 3e-8, 1e-7]
        elif args.parameter == "migration_rate":
            values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        
        # Run analysis for the selected parameter
        results = test_detection_power(args.parameter, values, 
                                    n_replicates=args.n_replicates,
                                    output_dir=args.output_dir)
        stats = calculate_roc_statistics(results, args.parameter, values)
        plot_detection_threshold(stats, args.parameter, output_dir=args.output_dir)
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    logging.info(f"Analysis completed in {elapsed_time:.2f} seconds")
#!/usr/bin/env python3
"""
Parameter Sweep Analysis for Introgression vs ILS Detection
This script creates a 2D parameter grid to analyze how different combinations
of parameters affect our ability to distinguish introgression from ILS.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import product
from matplotlib.colors import LinearSegmentedColormap

# Import your simulation module (adjust path as needed)
try:
    from enhanced_genetics_pipeline import main as run_simulation
except ImportError:
    logging.warning("Could not import simulation module. Running in analysis-only mode.")
    run_simulation = None

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("parameter_sweep.log"),
                        logging.StreamHandler()
                    ])

def run_2d_parameter_sweep(param1_name, param1_values, param2_name, param2_values, 
                          base_config, output_dir):
    """
    Run simulations with a 2D grid of parameter combinations
    
    Parameters:
    -----------
    param1_name : str
        Name of first parameter to sweep
    param1_values : list
        Values for first parameter
    param2_name : str
        Name of second parameter to sweep
    param2_values : list
        Values for second parameter
    base_config : dict
        Base configuration for simulations
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    pandas.DataFrame
        Results of parameter sweep
    """
    if run_simulation is None:
        logging.error("Simulation module not available. Cannot run parameter sweep.")
        return None
    
    # Create results structure
    results = []
    
    # Loop through parameter combinations
    for val1, val2 in product(param1_values, param2_values):
        # Update configuration with parameter values
        config = base_config.copy()
        
        # Set simulation parameters based on parameter names
        if param1_name == "introgression_proportion":
            config["introgression_proportion_range"] = [val1, val1]
        elif param1_name == "introgression_time":
            config["introgression_time_range"] = [val1, val1]
        elif param1_name == "recombination_rate":
            config["recombination_rate_range"] = [val1, val1]
        elif param1_name == "mutation_rate":
            config["mutation_rate_range"] = [val1, val1]
        
        if param2_name == "introgression_proportion":
            config["introgression_proportion_range"] = [val2, val2]
        elif param2_name == "introgression_time":
            config["introgression_time_range"] = [val2, val2]
        elif param2_name == "recombination_rate":
            config["recombination_rate_range"] = [val2, val2]
        elif param2_name == "mutation_rate":
            config["mutation_rate_range"] = [val2, val2]
        
        # For introgression parameters, ensure scenarios include introgression
        if "introgression" in param1_name or "introgression" in param2_name:
            config["scenario_weights"] = {
                'base': 0.5,
                'rapid_radiation': 0.0,
                'bottleneck': 0.0,
                'continuous_migration': 0.5
            }
        
        # Set output prefix for this run
        run_id = f"{param1_name}_{val1}_{param2_name}_{val2}"
        config["output_prefix"] = os.path.join(output_dir, f"2d_sweep_{run_id}")
        config["num_simulations"] = 50  # Smaller number per combination
        
        # Save configuration
        config_path = os.path.join(output_dir, f"config_{run_id}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run simulation
        logging.info(f"Running simulation with {param1_name}={val1}, {param2_name}={val2}")
        try:
            result = run_simulation(config_file=config_path)
            
            # Process results
            if isinstance(result, dict) and not result.get('error'):
                # Load detailed results
                results_file = f"{config['output_prefix']}_results.csv"
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    
                    # Calculate metrics for this parameter combination
                    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
                    
                    # Calculate AUC for D-statistic if possible
                    d_stat_auc = None
                    if 'd_stat' in df.columns and df['d_stat'].notna().sum() > 10:
                        valid_idx = df[['d_stat', 'has_introgression']].dropna().index
                        if len(valid_idx) > 10:
                            y_true = df.loc[valid_idx, 'has_introgression']
                            fpr, tpr, _ = roc_curve(y_true, df.loc[valid_idx, 'd_stat'])
                            d_stat_auc = auc(fpr, tpr)
                    
                    # Store parameter combination and results
                    sweep_result = {
                        param1_name: val1,
                        param2_name: val2,
                        'avg_d_stat': df['d_stat'].mean() if 'd_stat' in df.columns else None,
                        'avg_fst': df['fst'].mean() if 'fst' in df.columns else None,
                        'd_stat_auc': d_stat_auc,
                        'num_samples': len(df)
                    }
                    results.append(sweep_result)
                else:
                    logging.error(f"Results file not found: {results_file}")
            else:
                logging.error(f"Simulation failed for {param1_name}={val1}, {param2_name}={val2}")
        except Exception as e:
            logging.error(f"Error running simulation: {str(e)}")
    
    # Combine results
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(
            os.path.join(output_dir, f"2d_sweep_{param1_name}_{param2_name}_results.csv"),
            index=False
        )
        
        return results_df
    
    return None

def plot_2d_parameter_grid(results_df, output_dir):
    """
    Create heatmaps and surface plots from 2D parameter sweep results
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from 2D parameter sweep
    output_dir : str
        Directory to save visualizations
    """
    if results_df is None or len(results_df) == 0:
        logging.error("No results to plot")
        return
    
    # Extract parameter names
    param_cols = [col for col in results_df.columns 
                 if col not in ['avg_d_stat', 'avg_fst', 'd_stat_auc', 'num_samples']]
    
    if len(param_cols) != 2:
        logging.error(f"Expected 2 parameter columns, found {len(param_cols)}")
        return
    
    param1_name, param2_name = param_cols
    
    # Create custom colormap for AUC (red = poor, green = good)
    colors = [(0.8, 0.2, 0.2), (0.8, 0.8, 0.2), (0.2, 0.8, 0.2)]
    cm = LinearSegmentedColormap.from_list('auc_cmap', colors, N=100)
    
    # AUC heatmap
    if 'd_stat_auc' in results_df.columns and results_df['d_stat_auc'].notna().sum() > 0:
        plt.figure(figsize=(10, 8))
        
        # Pivot data for heatmap
        pivot_data = results_df.pivot(
            index=param1_name, columns=param2_name, values='d_stat_auc'
        )
        
        # Plot heatmap
        ax = sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.2f', 
            cmap=cm,
            vmin=0.5, 
            vmax=1.0,
            linewidths=0.5
        )
        
        plt.title(f'D-statistic AUC for Detecting Introgression\nVarying {param1_name} and {param2_name}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"auc_heatmap_{param1_name}_{param2_name}.png"),
            dpi=300
        )
        plt.close()
        
        # Create 3D surface plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = pivot_data.columns
        y = pivot_data.index
        X, Y = np.meshgrid(x, y)
        Z = pivot_data.values
        
        # Create surface plot
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap='viridis',
            linewidth=0,
            antialiased=True
        )
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_zlabel('AUC Score')
        
        # Set title
        ax.set_title(f'D-statistic AUC Surface\nVarying {param1_name} and {param2_name}')
        
        # Add horizontal plane at AUC=0.7 (threshold for good discrimination)
        threshold = 0.7
        x_grid, y_grid = np.meshgrid(
            np.linspace(min(x), max(x), 10),
            np.linspace(min(y), max(y), 10)
        )
        z_grid = np.full_like(x_grid, threshold)
        ax.plot_surface(x_grid, y_grid, z_grid, color='r', alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"auc_surface_{param1_name}_{param2_name}.png"),
            dpi=300
        )
        plt.close()
    
    # D-statistic by parameter combination
    if 'avg_d_stat' in results_df.columns and results_df['avg_d_stat'].notna().sum() > 0:
        plt.figure(figsize=(10, 8))
        
        # Pivot data
        pivot_data = results_df.pivot(
            index=param1_name, columns=param2_name, values='avg_d_stat'
        )
        
        # Plot heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            linewidths=0.5
        )
        
        plt.title(f'Average D-statistic\nVarying {param1_name} and {param2_name}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"d_stat_heatmap_{param1_name}_{param2_name}.png"),
            dpi=300
        )
        plt.close()
    
    # Create detection threshold visualization
    if 'd_stat_auc' in results_df.columns and results_df['d_stat_auc'].notna().sum() > 0:
        plt.figure(figsize=(10, 8))
        
        # Pivot data for binary detection threshold (AUC > 0.7 is "good")
        pivot_data = results_df.pivot(
            index=param1_name, columns=param2_name, values='d_stat_auc'
        )
        binary_detection = pivot_data > 0.7
        
        # Plot heatmap
        sns.heatmap(
            binary_detection.astype(int), 
            annot=pivot_data.round(2), 
            fmt='.2f', 
            cmap=['white', 'lightgreen'],
            linewidths=0.5,
            cbar_kws={'label': 'Reliable Detection (AUC > 0.7)'}
        )
        
        plt.title(f'Parameter Regions for Reliable Introgression Detection\nVarying {param1_name} and {param2_name}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"detection_threshold_{param1_name}_{param2_name}.png"),
            dpi=300
        )
        plt.close()
        
        # Save detection threshold recommendation
        with open(os.path.join(output_dir, f"detection_recommendations_{param1_name}_{param2_name}.txt"), 'w') as f:
            f.write(f"# Detection Recommendations for {param1_name} and {param2_name}\n\n")
            
            # Find optimal parameter combinations
            reliable_detection = results_df[results_df['d_stat_auc'] > 0.7]
            if len(reliable_detection) > 0:
                # Find best combination
                best_combo = reliable_detection.loc[reliable_detection['d_stat_auc'].idxmax()]
                
                f.write("## Recommended Parameter Ranges for Reliable Detection\n\n")
                f.write(f"For reliable distinction between introgression and ILS (AUC > 0.7):\n\n")
                
                # Extract parameter ranges with good detection
                good_param1_values = reliable_detection[param1_name].unique()
                good_param2_values = reliable_detection[param2_name].unique()
                
                f.write(f"- {param1_name}: {min(good_param1_values)} to {max(good_param1_values)}\n")
                f.write(f"- {param2_name}: {min(good_param2_values)} to {max(good_param2_values)}\n\n")
                
                f.write(f"## Optimal Parameter Combination\n\n")
                f.write(f"The optimal parameter combination (AUC = {best_combo['d_stat_auc']:.2f}):\n\n")
                f.write(f"- {param1_name}: {best_combo[param1_name]}\n")
                f.write(f"- {param2_name}: {best_combo[param2_name]}\n\n")
            else:
                f.write("No parameter combinations provided reliable detection (AUC > 0.7).\n")
                
                # Find best combination anyway
                best_combo = results_df.loc[results_df['d_stat_auc'].idxmax()]
                f.write(f"\nThe best available combination (AUC = {best_combo['d_stat_auc']:.2f}):\n\n")
                f.write(f"- {param1_name}: {best_combo[param1_name]}\n")
                f.write(f"- {param2_name}: {best_combo[param2_name]}\n\n")
            
            f.write("## Interpretation\n\n")
            
            # Interpret patterns in the data
            if param1_name == "introgression_time" or param2_name == "introgression_time":
                time_col = param1_name if param1_name == "introgression_time" else param2_name
                other_col = param2_name if param1_name == "introgression_time" else param1_name
                
                # Check correlation between time and AUC
                time_corr = results_df[[time_col, 'd_stat_auc']].corr().iloc[0, 1]
                
                if time_corr < -0.3:
                    f.write(f"Introgression detection becomes more difficult with increasing time since the " +
                           f"introgression event (correlation = {time_corr:.2f}). Recent introgression " +
                           f"events are easier to distinguish from ILS.\n\n")
                elif time_corr > 0.3:
                    f.write(f"Interestingly, introgression detection improves with increasing time " +
                           f"in this parameter range (correlation = {time_corr:.2f}). This may be related " +
                           f"to population-specific factors or the spread of introgressed alleles.\n\n")
                else:
                    f.write(f"Introgression time has a weak effect on detection reliability " +
                           f"in this parameter range (correlation = {time_corr:.2f}).\n\n")
            
            if param1_name == "introgression_proportion" or param2_name == "introgression_proportion":
                prop_col = param1_name if param1_name == "introgression_proportion" else param2_name
                other_col = param2_name if param1_name == "introgression_proportion" else param1_name
                
                # Check correlation between proportion and AUC
                prop_corr = results_df[[prop_col, 'd_stat_auc']].corr().iloc[0, 1]
                
                if prop_corr > 0.3:
                    f.write(f"Detection power increases with higher introgression proportion " +
                           f"(correlation = {prop_corr:.2f}). This indicates that larger amounts of " +
                           f"gene flow are easier to distinguish from ILS, as expected.\n\n")
                elif prop_corr < -0.3:
                    f.write(f"Unexpectedly, detection power decreases with higher introgression proportion " +
                           f"in this parameter range (correlation = {prop_corr:.2f}). This unusual pattern " +
                           f"may warrant further investigation.\n\n")
                else:
                    f.write(f"Introgression proportion has a weak effect on detection reliability " +
                           f"in this parameter range (correlation = {prop_corr:.2f}).\n\n")
            
            f.write("## Recommendations for Future Analyses\n\n")
            
            # Make recommendations based on patterns
            if 'introgression_time' in [param1_name, param2_name] and 'introgression_proportion' in [param1_name, param2_name]:
                f.write("When analyzing real genetic data:\n\n")
                
                if reliable_detection.empty:
                    f.write("1. Be cautious when interpreting D-statistics as evidence of introgression, " +
                           "as this analysis indicates limited discriminatory power in the tested parameter space.\n")
                    f.write("2. Consider using multiple complementary methods and incorporate additional evidence " +
                           "such as branch length distributions or haplotype block analysis.\n")
                else:
                    f.write("1. Higher confidence can be placed in introgression detection results when the " +
                           f"suspected parameters fall within the green region of the detection threshold map.\n")
                    f.write("2. Outside these parameter regions, consider supporting D-statistic evidence " +
                           "with additional analyses before drawing firm conclusions.\n")
            else:
                f.write("Further parameter exploration may help define the full range of conditions " +
                       "where introgression can be reliably distinguished from ILS.")

def analyze_sweep_results(results_file, output_dir):
    """
    Analyze existing parameter sweep results
    
    Parameters:
    -----------
    results_file : str
        Path to parameter sweep results CSV
    output_dir : str
        Directory to save analysis
    """
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        return
    
    # Load results
    results_df = pd.read_csv(results_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_2d_parameter_grid(results_df, output_dir)
    
    logging.info(f"Analysis of {results_file} completed. Results saved to {output_dir}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run parameter sweep for introgression/ILS detection")
    parser.add_argument("--output_dir", default="param_sweep_results", help="Output directory")
    parser.add_argument("--config", help="Base configuration file path (JSON)")
    parser.add_argument("--analyze_only", help="Analyze existing results file instead of running new simulations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze existing results if requested
    if args.analyze_only:
        logging.info(f"Analyzing existing results: {args.analyze_only}")
        analyze_sweep_results(args.analyze_only, args.output_dir)
        return
    
    # Load base configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    else:
        # Default configuration
        base_config = {
            'num_simulations': 50,
            'genome_length': 1e5,
            'max_workers': min(os.cpu_count(), 4),
            'save_trees': False,
            'scenario_weights': {
                'base': 0.5,
                'continuous_migration': 0.5
            }
        }
    
    # Define parameter ranges for introgression time vs proportion sweep
    introgression_times = [500, 1000, 1500, 2000, 2500]
    introgression_proportions = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Run 2D parameter sweep
    results_df = run_2d_parameter_sweep(
        "introgression_time", introgression_times,
        "introgression_proportion", introgression_proportions,
        base_config, args.output_dir
    )
    
    # Generate plots if sweep was successful
    if results_df is not None:
        plot_2d_parameter_grid(results_df, args.output_dir)
    
    logging.info("Parameter sweep completed")

if __name__ == "__main__":
    main()
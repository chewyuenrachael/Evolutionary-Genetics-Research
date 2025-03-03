#!/usr/bin/env python3
"""
Parameter Threshold Analysis Script

This script creates visualizations to identify parameter thresholds (migration rates,
recombination rates, etc.) that allow reliable detection of introgression using 
combinations of D-statistics and FST.

Usage:
    python parameter_threshold_analysis.py --input_file results/simulation_results.csv --output_dir threshold_plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

def plot_d_stat_vs_fst_by_migration_rate(df, output_dir):
    """Create scatter plot of D-statistic vs FST colored by migration rate"""
    # Check if migration_rate column exists
    if 'migration_rate' not in df.columns or df['migration_rate'].isna().all():
        print("No migration rate data available for plot")
        return
    
    # Filter to only rows with migration rate data
    plot_df = df.dropna(subset=['d_stat', 'fst', 'migration_rate'])
    
    if len(plot_df) < 10:
        print("Not enough data points with migration rate information")
        return
    
    # Create log of migration rate for better visualization
    plot_df['log_migration_rate'] = np.log10(plot_df['migration_rate'])
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        plot_df['d_stat'],
        plot_df['fst'],
        c=plot_df['log_migration_rate'],
        cmap='viridis',
        alpha=0.8,
        s=70
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log10(Migration Rate)')
    
    # Add labels
    plt.title('D-statistic vs FST by Migration Rate')
    plt.xlabel('D-statistic')
    plt.ylabel('FST')
    plt.grid(alpha=0.3)
    
    # Add a best fit line
    m, b = np.polyfit(plot_df['d_stat'], plot_df['fst'], 1)
    plt.plot(plot_df['d_stat'], m*plot_df['d_stat'] + b, 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_migration_rate.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by migration rate")
    
    # Create categorical versions with migration rate bins
    # Bin the migration rates into categories for easier visualization
    migration_bins = np.logspace(
        np.floor(np.log10(plot_df['migration_rate'].min())),
        np.ceil(np.log10(plot_df['migration_rate'].max())),
        5
    )
    
    migration_labels = [f"{migration_bins[i]:.2e} - {migration_bins[i+1]:.2e}" 
                        for i in range(len(migration_bins)-1)]
    
    plot_df['migration_bin'] = pd.cut(
        plot_df['migration_rate'], 
        bins=migration_bins, 
        labels=migration_labels
    )
    
    # Create facet plot
    plt.figure(figsize=(18, 10))
    g = sns.FacetGrid(plot_df, col="migration_bin", col_wrap=2, height=4)
    g.map_dataframe(sns.scatterplot, x="d_stat", y="fst", hue="scenario", alpha=0.7)
    g.add_legend()
    g.set_axis_labels("D-statistic", "FST")
    g.set_titles("Migration Rate: {col_name}")
    g.fig.suptitle('D-statistic vs FST by Migration Rate Bins', y=1.02, fontsize=16)
    g.fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_migration_rate_bins.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by migration rate bins")

def plot_d_stat_vs_fst_by_recombination_rate(df, output_dir):
    """Create scatter plot of D-statistic vs FST colored by recombination rate"""
    # Filter to only rows with needed data
    plot_df = df.dropna(subset=['d_stat', 'fst', 'recombination_rate'])
    
    if len(plot_df) < 10:
        print("Not enough data points for recombination rate analysis")
        return
    
    # Create log of recombination rate for better visualization
    plot_df['log_recombination_rate'] = np.log10(plot_df['recombination_rate'])
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        plot_df['d_stat'],
        plot_df['fst'],
        c=plot_df['log_recombination_rate'],
        cmap='plasma',
        alpha=0.8,
        s=70
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log10(Recombination Rate)')
    
    # Add labels
    plt.title('D-statistic vs FST by Recombination Rate')
    plt.xlabel('D-statistic')
    plt.ylabel('FST')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_recombination_rate.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by recombination rate")
    
    # Create categorical versions with recombination rate bins
    recombination_bins = np.logspace(
        np.floor(np.log10(plot_df['recombination_rate'].min())),
        np.ceil(np.log10(plot_df['recombination_rate'].max())),
        5
    )
    
    recombination_labels = [f"{recombination_bins[i]:.2e} - {recombination_bins[i+1]:.2e}" 
                            for i in range(len(recombination_bins)-1)]
    
    plot_df['recombination_bin'] = pd.cut(
        plot_df['recombination_rate'], 
        bins=recombination_bins, 
        labels=recombination_labels
    )
    
    # Create facet plot
    plt.figure(figsize=(18, 10))
    g = sns.FacetGrid(plot_df, col="recombination_bin", col_wrap=2, height=4)
    g.map_dataframe(sns.scatterplot, x="d_stat", y="fst", hue="scenario", alpha=0.7)
    g.add_legend()
    g.set_axis_labels("D-statistic", "FST")
    g.set_titles("Recombination Rate: {col_name}")
    g.fig.suptitle('D-statistic vs FST by Recombination Rate Bins', y=1.02, fontsize=16)
    g.fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_recombination_rate_bins.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by recombination rate bins")

def plot_d_stat_vs_fst_by_introgression_proportion(df, output_dir):
    """Create scatter plot of D-statistic vs FST colored by introgression proportion"""
    # Filter to only rows with needed data
    plot_df = df.dropna(subset=['d_stat', 'fst', 'introgression_proportion'])
    
    if len(plot_df) < 10:
        print("Not enough data points for introgression proportion analysis")
        return
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        plot_df['d_stat'],
        plot_df['fst'],
        c=plot_df['introgression_proportion'],
        cmap='YlOrRd',
        alpha=0.8,
        s=70
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Introgression Proportion')
    
    # Add labels
    plt.title('D-statistic vs FST by Introgression Proportion')
    plt.xlabel('D-statistic')
    plt.ylabel('FST')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_introgression_proportion.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by introgression proportion")
    
    # Create categorical versions with introgression proportion bins
    intro_bins = np.linspace(
        plot_df['introgression_proportion'].min(),
        plot_df['introgression_proportion'].max(),
        5
    )
    
    intro_labels = [f"{intro_bins[i]:.2f} - {intro_bins[i+1]:.2f}" 
                    for i in range(len(intro_bins)-1)]
    
    plot_df['introgression_bin'] = pd.cut(
        plot_df['introgression_proportion'], 
        bins=intro_bins, 
        labels=intro_labels
    )
    
    # Create facet plot
    plt.figure(figsize=(18, 10))
    g = sns.FacetGrid(plot_df, col="introgression_bin", col_wrap=2, height=4)
    g.map_dataframe(sns.scatterplot, x="d_stat", y="fst", hue="scenario", alpha=0.7)
    g.add_legend()
    g.set_axis_labels("D-statistic", "FST")
    g.set_titles("Introgression Proportion: {col_name}")
    g.fig.suptitle('D-statistic vs FST by Introgression Proportion Bins', y=1.02, fontsize=16)
    g.fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_fst_by_introgression_proportion_bins.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST by introgression proportion bins")

def calculate_detection_power_by_parameter(df, param_name, output_dir):
    """Calculate detection power (AUC) at different parameter values"""
    if param_name not in df.columns or df[param_name].isna().all():
        print(f"No {param_name} data available for analysis")
        return
    
    # Create has_introgression label (adjust as needed based on your scenarios)
    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    # Create bins for the parameter
    if param_name in ['recombination_rate', 'mutation_rate', 'migration_rate']:
        # For rates, use logarithmic bins
        param_min = np.floor(np.log10(df[param_name].dropna().min()))
        param_max = np.ceil(np.log10(df[param_name].dropna().max()))
        bins = np.logspace(param_min, param_max, 8)  # 7 bins
    else:
        # For proportions and times, use linear bins
        bins = np.linspace(df[param_name].dropna().min(), df[param_name].dropna().max(), 8)
    
    # Create bin labels
    if param_name in ['recombination_rate', 'mutation_rate', 'migration_rate']:
        bin_labels = [f"{bins[i]:.2e} - {bins[i+1]:.2e}" for i in range(len(bins)-1)]
    else:
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
    
    # Bin the parameter values
    df[f'{param_name}_bin'] = pd.cut(df[param_name], bins=bins, labels=bin_labels)
    
    # Calculate AUC for each bin and each metric
    metrics = ['d_stat', 'fst', 'mean_internal_branch', 'std_internal_branch']
    results = []
    
    for bin_label in bin_labels:
        bin_df = df[df[f'{param_name}_bin'] == bin_label]
        if len(bin_df) < 10:
            continue
            
        for metric in metrics:
            if metric not in bin_df.columns or bin_df[metric].isna().all():
                continue
                
            # For metrics where lower values indicate introgression
            if metric in ['fst', 'mean_internal_branch']:
                values = -bin_df[metric]
            else:
                values = bin_df[metric]
                
            # Calculate AUC
            try:
                valid_idx = bin_df[[metric, 'has_introgression']].dropna().index
                if len(valid_idx) < 5:
                    continue
                    
                fpr, tpr, _ = roc_curve(bin_df.loc[valid_idx, 'has_introgression'], values.loc[valid_idx])
                metric_auc = auc(fpr, tpr)
                
                # Calculate median parameter value in this bin
                param_median = bin_df[param_name].median()
                
                # Store results
                results.append({
                    'bin': bin_label,
                    'metric': metric,
                    'auc': metric_auc,
                    f'{param_name}_median': param_median,
                    'sample_size': len(valid_idx)
                })
            except Exception as e:
                print(f"Error calculating AUC for {metric} in bin {bin_label}: {e}")
    
    if not results:
        print(f"No valid AUC calculations for {param_name}")
        return
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot AUC by parameter value
    plt.figure(figsize=(12, 8))
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric]
        if len(metric_data) < 2:
            continue
            
        plt.plot(
            metric_data[f'{param_name}_median'],
            metric_data['auc'],
            'o-',
            label=metric,
            linewidth=2,
            markersize=8
        )
    
    # Format x-axis for rates
    if param_name in ['recombination_rate', 'mutation_rate', 'migration_rate']:
        plt.xscale('log')
        plt.xlabel(f"{param_name.replace('_', ' ').title()} (log scale)")
    else:
        plt.xlabel(param_name.replace('_', ' ').title())
    
    plt.ylabel('AUC (Detection Power)')
    plt.title(f'Detection Power by {param_name.replace("_", " ").title()}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7)
    plt.text(results_df[f'{param_name}_median'].max() * 0.95, 0.72, 'Good Detection (AUC > 0.7)', 
             ha='right', va='bottom', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'detection_power_by_{param_name}.png'), dpi=300)
    plt.close()
    print(f"Created: Detection power by {param_name}")
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, f'detection_power_by_{param_name}.csv'), index=False)
    print(f"Saved: Detection power results for {param_name}")
    
    return results_df

def create_2d_decision_boundary(df, output_dir):
    """Create 2D decision boundary for introgression detection using D-stat and FST"""
    # Filter to only rows with needed data
    plot_df = df.dropna(subset=['d_stat', 'fst'])
    
    # Create has_introgression label
    plot_df['has_introgression'] = ~plot_df['scenario'].isin(['continuous_migration'])
    
    if len(plot_df) < 20:
        print("Not enough data points for decision boundary analysis")
        return
    
    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot the scatter points
    sns.scatterplot(
        data=plot_df,
        x='d_stat',
        y='fst',
        hue='has_introgression',
        palette={True: 'green', False: 'red'},
        alpha=0.7,
        s=70
    )
    
    # Create a 2D grid for density estimation
    x_min, x_max = plot_df['d_stat'].min() - 0.1, plot_df['d_stat'].max() + 0.1
    y_min, y_max = plot_df['fst'].min() - 0.1, plot_df['fst'].max() + 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # Estimate density for introgression and non-introgression points
    introgression_points = plot_df[plot_df['has_introgression']][['d_stat', 'fst']].values
    non_introgression_points = plot_df[~plot_df['has_introgression']][['d_stat', 'fst']].values
    
    # Only proceed if we have enough points in each category
    if len(introgression_points) < 5 or len(non_introgression_points) < 5:
        print("Not enough points in each category for density estimation")
        return
    
    try:
        # Fit kernel density models
        kde_introgression = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde_introgression.fit(introgression_points)
        
        kde_non_introgression = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde_non_introgression.fit(non_introgression_points)
        
        # Calculate densities on the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        introgression_density = np.exp(kde_introgression.score_samples(grid_points))
        non_introgression_density = np.exp(kde_non_introgression.score_samples(grid_points))
        
        # Calculate probability of introgression
        p_introgression = introgression_density / (introgression_density + non_introgression_density)
        
        # Reshape the probability array to match the grid
        p_introgression = p_introgression.reshape(xx.shape)
        
        # Create a custom colormap (white to green)
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            [(1, 1, 1, 0), (0.2, 0.8, 0.2, 0.6)],
            N=100
        )
        
        # Plot the contour
        contour = plt.contourf(xx, yy, p_introgression, levels=np.linspace(0.5, 1, 10), cmap=cmap, alpha=0.3)
        
        # Add a colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Probability of Introgression')
        
    except Exception as e:
        print(f"Error in density estimation: {e}")
    
    # Add title and labels
    plt.title('D-statistic vs FST: Introgression Decision Boundary')
    plt.xlabel('D-statistic')
    plt.ylabel('FST')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'introgression_decision_boundary.png'), dpi=300)
    plt.close()
    print("Created: Introgression decision boundary")

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parameter Threshold Analysis Script")
    parser.add_argument("--input_file", required=True, help="Path to simulation results CSV file")
    parser.add_argument("--output_dir", default="threshold_plots", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Load data
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} rows of data")
    
    # Print scenario counts
    if 'scenario' in df.columns:
        scenario_counts = df['scenario'].value_counts()
        print("\nScenario counts:")
        for scenario, count in scenario_counts.items():
            print(f"  {scenario}: {count}")
    
    # Generate plots
    plot_d_stat_vs_fst_by_migration_rate(df, args.output_dir)
    plot_d_stat_vs_fst_by_recombination_rate(df, args.output_dir)
    plot_d_stat_vs_fst_by_introgression_proportion(df, args.output_dir)
    
    # Calculate detection power by parameter
    calculate_detection_power_by_parameter(df, 'migration_rate', args.output_dir)
    calculate_detection_power_by_parameter(df, 'recombination_rate', args.output_dir)
    calculate_detection_power_by_parameter(df, 'introgression_proportion', args.output_dir)
    calculate_detection_power_by_parameter(df, 'introgression_time', args.output_dir)
    
    # Create decision boundary
    create_2d_decision_boundary(df, args.output_dir)
    
    print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
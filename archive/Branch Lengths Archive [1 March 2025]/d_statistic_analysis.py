#!/usr/bin/env python3
"""
D-statistic Visualization Script

This script creates comprehensive visualizations of D-statistics from simulation results,
comparing them with branch length metrics to evaluate their effectiveness in distinguishing
introgression from incomplete lineage sorting (ILS).

Usage:
    python d_statistic_analysis.py --input_file results/simulation_results.csv --output_dir d_stat_plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

def plot_d_stat_by_scenario(df, output_dir):
    """Create violin plot of D-statistics by scenario"""
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='scenario', y='d_stat', data=df, palette='viridis')
    plt.title('D-statistic Distribution by Evolutionary Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('D-statistic')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_by_scenario.png'), dpi=300)
    plt.close()
    print("Created: D-statistic distribution by scenario")

def plot_d_stat_vs_introgression_proportion(df, output_dir):
    """Create scatter plot of D-statistic vs introgression proportion"""
    # Only include rows where introgression_proportion is not NA
    plot_df = df.dropna(subset=['introgression_proportion', 'd_stat'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df, 
        x='introgression_proportion', 
        y='d_stat',
        hue='scenario',
        palette='viridis',
        alpha=0.7
    )
    plt.title('D-statistic vs. Introgression Proportion')
    plt.xlabel('Introgression Proportion')
    plt.ylabel('D-statistic')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_introgression_proportion.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs introgression proportion")

def plot_d_stat_roc_curve(df, output_dir):
    """Create ROC curve for D-statistic detection power"""
    # Create binary label: True for introgression scenarios (not continuous_migration)
    # Note: Adjust this based on your actual scenario definitions
    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(df['has_introgression'], df['d_stat'])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for D-statistic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_roc_curve.png'), dpi=300)
    plt.close()
    print(f"Created: ROC curve for D-statistic (AUC = {roc_auc:.2f})")
    
    # Create precision-recall curve for more balanced evaluation
    precision, recall, _ = precision_recall_curve(df['has_introgression'], df['d_stat'])
    avg_precision = average_precision_score(df['has_introgression'], df['d_stat'])
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for D-statistic')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_precision_recall.png'), dpi=300)
    plt.close()
    print(f"Created: Precision-Recall curve for D-statistic (AP = {avg_precision:.2f})")

def plot_d_stat_vs_branch_metrics(df, output_dir):
    """Create scatter plots comparing D-stat with branch length metrics"""
    # D-stat vs std_internal_branch
    plt.figure(figsize=(10, 6))
    plot_df = df.dropna(subset=['std_internal_branch', 'd_stat'])
    sns.scatterplot(
        data=plot_df,
        x='d_stat',
        y='std_internal_branch',
        hue='scenario',
        palette='viridis',
        alpha=0.7
    )
    plt.title('D-statistic vs. Branch Length Variability')
    plt.xlabel('D-statistic')
    plt.ylabel('Standard Deviation of Branch Lengths')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_branch_std.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs branch length variability")
    
    # D-stat vs mean_internal_branch
    plt.figure(figsize=(10, 6))
    plot_df = df.dropna(subset=['mean_internal_branch', 'd_stat'])
    sns.scatterplot(
        data=plot_df,
        x='d_stat',
        y='mean_internal_branch',
        hue='scenario',
        palette='viridis',
        alpha=0.7
    )
    plt.title('D-statistic vs. Mean Branch Length')
    plt.xlabel('D-statistic')
    plt.ylabel('Mean Internal Branch Length')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_vs_branch_mean.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs mean branch length")

def plot_d_stat_density(df, output_dir):
    """Create density plot of D-statistics by scenario"""
    plt.figure(figsize=(10, 6))
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]['d_stat'].dropna()
        sns.kdeplot(scenario_data, label=scenario, fill=True, alpha=0.3)

    plt.title('D-statistic Density by Evolutionary Scenario')
    plt.xlabel('D-statistic')
    plt.ylabel('Density')
    plt.legend(title='Scenario')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'd_stat_density.png'), dpi=300)
    plt.close()
    print("Created: D-statistic density by scenario")

def plot_comparative_detection_power(df, output_dir):
    """Compare detection power of D-statistics vs. branch length metrics"""
    # Create has_introgression label
    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    # Create plot for comparison of metrics
    metrics = ['d_stat', 'std_internal_branch', 'mean_internal_branch', 'fst']
    plt.figure(figsize=(10, 8))
    
    for metric in metrics:
        # Skip metrics with too many missing values
        if df[metric].isna().sum() > len(df) * 0.5:
            print(f"Skipping {metric} due to too many missing values")
            continue
            
        # Handle negative correlation metrics (where smaller values indicate introgression)
        if metric == 'mean_internal_branch':
            values = -df[metric]  # Invert if negatively correlated with introgression
        else:
            values = df[metric]
            
        # Calculate ROC curve
        valid_idx = df[[metric, 'has_introgression']].dropna().index
        if len(valid_idx) < 10:
            print(f"Skipping {metric} due to insufficient data")
            continue
            
        fpr, tpr, _ = roc_curve(df.loc[valid_idx, 'has_introgression'], values.loc[valid_idx])
        metric_auc = auc(fpr, tpr)
        
        # Plot ROC curve for this metric
        plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {metric_auc:.2f})')
    
    # Add reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curves for Introgression Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_roc_curves.png'), dpi=300)
    plt.close()
    print("Created: Comparative ROC curves for detection metrics")

def create_summary_table(df, output_dir):
    """Create summary table of D-statistics by scenario"""
    # Calculate summary statistics by scenario
    summary = df.groupby('scenario')['d_stat'].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: x.quantile(0.25).round(3),
        lambda x: x.quantile(0.5).round(3),
        lambda x: x.quantile(0.75).round(3)
    ]).rename(columns={
        '<lambda_0>': 'q25',
        '<lambda_1>': 'median',
        '<lambda_2>': 'q75'
    }).reset_index()
    
    # Round numerical columns
    for col in summary.columns:
        if col != 'scenario' and summary[col].dtype in [np.float64, np.int64]:
            summary[col] = summary[col].round(3)
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, 'd_stat_summary.csv'), index=False)
    print("Created: D-statistic summary table by scenario")
    
    # Return summary for inspection
    return summary

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="D-statistic Visualization Script")
    parser.add_argument("--input_file", required=True, help="Path to simulation results CSV file")
    parser.add_argument("--output_dir", default="d_stat_plots", help="Directory to save visualizations")
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
    
    # Generate all plots
    plot_d_stat_by_scenario(df, args.output_dir)
    plot_d_stat_vs_introgression_proportion(df, args.output_dir)
    plot_d_stat_roc_curve(df, args.output_dir)
    plot_d_stat_vs_branch_metrics(df, args.output_dir)
    plot_d_stat_density(df, args.output_dir)
    plot_comparative_detection_power(df, args.output_dir)
    
    # Create summary table
    summary = create_summary_table(df, args.output_dir)
    print("\nD-statistic summary by scenario:")
    print(summary)
    
    print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
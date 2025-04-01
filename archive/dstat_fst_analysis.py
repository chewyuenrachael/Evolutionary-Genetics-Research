#!/usr/bin/env python3
"""
D-statistic and FST Analysis Script

This script creates comprehensive visualizations comparing D-statistics and FST values
across different evolutionary scenarios (base, rapid_radiation, bottleneck, continuous_migration)
to help distinguish introgression from incomplete lineage sorting (ILS).

Usage:
    python dstat_fst_analysis.py --input_file results/simulation_results.csv --output_dir analysis_output
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

def plot_dstat_by_scenario(df, output_dir):
    """Create density plot of D-statistics by scenario"""
    plt.figure(figsize=(12, 8))
    
    # Set custom color palette for scenarios to match the images
    scenario_palette = {
        'continuous_migration': '#3498db',  # Blue
        'rapid_radiation': '#f39c12',       # Orange
        'base': '#2ecc71',                  # Green
        'bottleneck': '#e74c3c'             # Red
    }
    
    # Create the density plots (KDE) with custom colors
    for scenario, color in scenario_palette.items():
        scenario_data = df[df['scenario'] == scenario]['d_stat'].dropna()
        if len(scenario_data) > 0:
            sns.kdeplot(
                scenario_data,
                label=scenario,
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=2
            )
    
    plt.title('D-statistic Density by Evolutionary Scenario')
    plt.xlabel('D-statistic')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Scenario')
    
    # Add horizontal line at y=0 for reference
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set x-axis limits to match the image (-2 to 2)
    plt.xlim(-2.0, 2.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dstat_by_scenario.png'), dpi=300)
    plt.close()
    print("Created: D-statistic density plot by scenario")
    
    # Save summary statistics to CSV
    summary = df.groupby('scenario')['d_stat'].agg(['count', 'mean', 'std', 'min', 'max', 'median'])
    summary.to_csv(os.path.join(output_dir, 'dstat_summary_by_scenario.csv'))
    print("Saved: D-statistic summary statistics by scenario")

def plot_fst_by_scenario(df, output_dir):
    """Create density plot of FST values by scenario"""
    plt.figure(figsize=(12, 8))
    
    # Set custom color palette for scenarios to match the images
    scenario_palette = {
        'continuous_migration': '#3498db',  # Blue
        'rapid_radiation': '#f39c12',       # Orange
        'base': '#2ecc71',                  # Green
        'bottleneck': '#e74c3c'             # Red
    }
    
    # Create the density plots (KDE) with custom colors
    for scenario, color in scenario_palette.items():
        scenario_data = df[df['scenario'] == scenario]['fst'].dropna()
        if len(scenario_data) > 0:
            sns.kdeplot(
                scenario_data,
                label=scenario,
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=2
            )
    
    plt.title('FST Density by Evolutionary Scenario')
    plt.xlabel('FST')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Scenario')
    
    # Set x-axis limits to match the image (-0.2 to 1.2)
    plt.xlim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fst_by_scenario.png'), dpi=300)
    plt.close()
    print("Created: FST density plot by scenario")
    
    # Save summary statistics to CSV
    summary = df.groupby('scenario')['fst'].agg(['count', 'mean', 'std', 'min', 'max', 'median'])
    summary.to_csv(os.path.join(output_dir, 'fst_summary_by_scenario.csv'))
    print("Saved: FST summary statistics by scenario")

def plot_dstat_vs_fst(df, output_dir):
    """Create scatter plot of D-statistic vs FST grouped by scenario"""
    plt.figure(figsize=(12, 8))
    
    # Set custom color palette for scenarios
    scenario_palette = {
        'base': '#3498db',            # Blue
        'rapid_radiation': '#e74c3c', # Red
        'bottleneck': '#2ecc71',      # Green
        'continuous_migration': '#9b59b6'  # Purple
    }
    
    # Filter data to only rows with both D-stat and FST values
    plot_df = df.dropna(subset=['d_stat', 'fst'])
    
    # Create scatter plot
    g = sns.scatterplot(
        data=plot_df,
        x='d_stat',
        y='fst',
        hue='scenario',
        palette=scenario_palette,
        alpha=0.7,
        s=70
    )
    
    # Add regression line for each scenario
    scenarios = plot_df['scenario'].unique()
    for scenario in scenarios:
        scenario_data = plot_df[plot_df['scenario'] == scenario]
        if len(scenario_data) > 5:  # Only add line if we have enough data points
            sns.regplot(
                x='d_stat',
                y='fst',
                data=scenario_data,
                scatter=False,
                color=scenario_palette.get(scenario, 'gray'),
                line_kws={'linestyle': '--', 'linewidth': 1}
            )
    
    # Calculate correlation for each scenario
    correlations = []
    for scenario in scenarios:
        scenario_data = plot_df[plot_df['scenario'] == scenario]
        if len(scenario_data) > 5:
            corr = scenario_data['d_stat'].corr(scenario_data['fst'])
            correlations.append({'scenario': scenario, 'correlation': corr})
    
    # Add correlation values to the legend
    if correlations:
        corr_df = pd.DataFrame(correlations)
        handles, labels = g.get_legend_handles_labels()
        new_labels = []
        for label in labels:
            if label in corr_df['scenario'].values:
                corr_val = corr_df[corr_df['scenario'] == label]['correlation'].values[0]
                new_labels.append(f"{label} (r = {corr_val:.2f})")
            else:
                new_labels.append(label)
        g.legend(handles=handles, labels=new_labels, title='Scenario')
    
    plt.title('D-statistic vs FST by Evolutionary Scenario')
    plt.xlabel('D-statistic')
    plt.ylabel('FST')
    plt.grid(alpha=0.3)
    
    # Add overall correlation
    overall_corr = plot_df['d_stat'].corr(plot_df['fst'])
    plt.figtext(0.5, 0.01, f'Overall Correlation: {overall_corr:.3f}', 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dstat_vs_fst.png'), dpi=300)
    plt.close()
    print("Created: D-statistic vs FST scatter plot")
    
    # Also create a faceted version (one plot per scenario)
    plt.figure(figsize=(16, 12))
    g = sns.FacetGrid(plot_df, col="scenario", col_wrap=2, height=5, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x="d_stat", y="fst", alpha=0.7)
    g.map_dataframe(sns.regplot, x="d_stat", y="fst", scatter=False, line_kws={'color': 'red'})
    
    # Add correlation to each subplot
    for ax, scenario in zip(g.axes, scenarios):
        scenario_data = plot_df[plot_df['scenario'] == scenario]
        if len(scenario_data) > 5:
            corr = scenario_data['d_stat'].corr(scenario_data['fst'])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', va='top')
    
    g.set_axis_labels("D-statistic", "FST")
    g.set_titles("Scenario: {col_name}")
    g.fig.suptitle('D-statistic vs FST Relationship by Scenario', y=1.02, fontsize=16)
    g.fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dstat_vs_fst_faceted.png'), dpi=300)
    plt.close()
    print("Created: Faceted D-statistic vs FST plots by scenario")

def plot_roc_curves(df, output_dir):
    """Create ROC curves for each scenario to evaluate detection power"""
    # Create a binary classification problem:
    # We'll consider "has_introgression" as positive class (where introgression is present)
    # Continuous migration is treated as ILS (no introgression)
    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    plt.figure(figsize=(10, 8))
    
    # Calculate overall ROC curves for each metric
    metrics = ['d_stat', 'fst']
    metric_labels = {'d_stat': 'D-statistic', 'fst': 'FST'}
    colors = {'d_stat': '#e74c3c', 'fst': '#3498db'}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # For FST, lower values suggest introgression, so we use negative values
        if metric == 'fst':
            y_score = -df[metric].values
        else:
            y_score = df[metric].values
            
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(df['has_introgression'], y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[metric], lw=2,
                 label=f'{metric_labels[metric]} (AUC = {roc_auc:.3f})')
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random Chance (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Introgression Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'introgression_roc_curves.png'), dpi=300)
    plt.close()
    print("Created: ROC curves for introgression detection")
    
    # Now create ROC curves for each scenario separately
    introgression_scenarios = [s for s in df['scenario'].unique() if s != 'continuous_migration']
    
    for scenario in introgression_scenarios:
        plt.figure(figsize=(10, 8))
        
        # Filter data to compare this scenario vs continuous_migration
        scenario_df = df[(df['scenario'] == scenario) | (df['scenario'] == 'continuous_migration')]
        scenario_label = f"'{scenario}' vs 'continuous_migration'"
        
        for metric in metrics:
            if metric not in scenario_df.columns:
                continue
                
            # For FST, lower values suggest introgression, so we use negative values
            if metric == 'fst':
                y_score = -scenario_df[metric].values
            else:
                y_score = scenario_df[metric].values
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(scenario_df['has_introgression'], y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[metric], lw=2,
                     label=f'{metric_labels[metric]} (AUC = {roc_auc:.3f})')
        
        # Plot random chance line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
                 label='Random Chance (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for Scenario: {scenario_label}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{scenario}.png'), dpi=300)
        plt.close()
        print(f"Created: ROC curve for scenario '{scenario}'")

def analyze_parameter_impacts(df, output_dir):
    """Analyze how different parameters affect D-statistic and FST values"""
    # Parameters to analyze
    parameters = ['recombination_rate', 'mutation_rate', 
                  'introgression_proportion', 'introgression_time']
    
    # Check which parameters are present in the data
    available_params = [p for p in parameters if p in df.columns]
    
    for param in available_params:
        plt.figure(figsize=(12, 10))
        
        # Create subplots for D-statistic and FST
        plt.subplot(2, 1, 1)
        sns.scatterplot(
            data=df,
            x=param,
            y='d_stat',
            hue='scenario',
            alpha=0.7
        )
        
        # Add a regression line
        sns.regplot(
            data=df,
            x=param,
            y='d_stat',
            scatter=False,
            color='black',
            line_kws={'linestyle': '--'}
        )
        
        # Calculate correlation
        correlation = df[[param, 'd_stat']].corr().iloc[0, 1]
        
        plt.title(f'D-statistic vs {param.replace("_", " ").title()} (r = {correlation:.3f})')
        plt.ylabel('D-statistic')
        
        # For rate parameters, use log scale
        if 'rate' in param:
            plt.xscale('log')
            
        plt.grid(True, alpha=0.3)
        
        # Second subplot for FST
        plt.subplot(2, 1, 2)
        sns.scatterplot(
            data=df,
            x=param,
            y='fst',
            hue='scenario',
            alpha=0.7
        )
        
        # Add a regression line
        sns.regplot(
            data=df,
            x=param,
            y='fst',
            scatter=False,
            color='black',
            line_kws={'linestyle': '--'}
        )
        
        # Calculate correlation
        correlation = df[[param, 'fst']].corr().iloc[0, 1]
        
        plt.title(f'FST vs {param.replace("_", " ").title()} (r = {correlation:.3f})')
        plt.xlabel(param.replace('_', ' ').title())
        plt.ylabel('FST')
        
        # For rate parameters, use log scale
        if 'rate' in param:
            plt.xscale('log')
            
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_impact_{param}.png'), dpi=300)
        plt.close()
        print(f"Created: Parameter impact analysis for {param}")

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="D-statistic and FST Analysis Script")
    parser.add_argument("--input_file", required=True, help="Path to simulation results CSV file")
    parser.add_argument("--output_dir", default="dstat_fst_analysis", help="Directory to save visualizations")
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
    
    # Plot D-statistic by scenario
    plot_dstat_by_scenario(df, args.output_dir)
    
    # Plot FST by scenario
    plot_fst_by_scenario(df, args.output_dir)
    
    # Plot D-statistic vs FST
    plot_dstat_vs_fst(df, args.output_dir)
    
    # Plot ROC curves
    plot_roc_curves(df, args.output_dir)
    
    # Analyze parameter impacts
    analyze_parameter_impacts(df, args.output_dir)
    
    print(f"\nAll analyses completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
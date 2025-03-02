#!/usr/bin/env python3
"""
Visualization Tools for Evolutionary Genetics Analysis
Supporting introgression vs ILS comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from Bio import Phylo
from io import StringIO
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set default styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="viridis")
SCENARIOS_COLORS = {
    'base': '#3498db',
    'rapid_radiation': '#e74c3c',
    'bottleneck': '#2ecc71',
    'continuous_migration': '#9b59b6'
}

def plot_parameter_space(df: pd.DataFrame, 
                         output_file: str = "parameter_space.png") -> None:
    """
    Create 2D kernel density plots for key parameter combinations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_file : str
        Output file path
    """
    plt.figure(figsize=(20, 15))
    
    # Define key parameter combinations
    param_combinations = [
        ('recombination_rate', 'fst', 'd_stat'),
        ('recombination_rate', 'introgression_time', 'd_stat'),
        ('mutation_rate', 'fst', 'd_stat'),
        ('introgression_proportion', 'fst', 'd_stat')
    ]
    
    for i, (x_param, y_param, color_param) in enumerate(param_combinations):
        # Skip if any params are missing
        if not all(p in df.columns for p in [x_param, y_param, color_param]):
            continue
            
        plt.subplot(2, 2, i+1)
        
        # Create scatter plot with color based on d_stat
        scatter = plt.scatter(
            df[x_param], 
            df[y_param],
            c=df[color_param], 
            cmap='viridis',
            alpha=0.7,
            s=50,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Add colorbar
        plt.colorbar(scatter, label=color_param.replace('_', ' ').title())
        
        # Add labels and title
        plt.xlabel(x_param.replace('_', ' ').title())
        plt.ylabel(y_param.replace('_', ' ').title())
        plt.title(f"{y_param.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}")
        
        # Scale axes - log scale for rates
        if 'rate' in x_param:
            plt.xscale('log')
        if 'rate' in y_param:
            plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Parameter space plot saved to {output_file}")

def plot_3d_parameter_space(df: pd.DataFrame, 
                           output_file: str = "parameter_space_3d.html") -> None:
    """
    Create interactive 3D plot of parameter space
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_file : str
        Output HTML file path
    """
    # Check if required columns exist
    if not all(col in df.columns for col in ['recombination_rate', 'introgression_time', 'fst', 'd_stat']):
        logging.warning("Required columns missing for 3D parameter space plot")
        return
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df['recombination_rate'],
            y=df['introgression_time'],
            z=df['fst'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['d_stat'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='D-statistic')
            ),
            text=[f"Scenario: {s}<br>D-stat: {d:.2f}<br>FST: {f:.2f}" 
                  for s, d, f in zip(df['scenario'], df['d_stat'], df['fst'])],
            hoverinfo='text'
        )
    ])
    
    # Customize layout
    fig.update_layout(
        title="3D Parameter Space Exploration",
        scene=dict(
            xaxis_title='Recombination Rate (log scale)',
            yaxis_title='Introgression Time',
            zaxis_title='FST',
            xaxis_type="log"
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        height=800
    )
    
    # Save as HTML for interactivity
    fig.write_html(output_file)
    logging.info(f"3D parameter space plot saved to {output_file}")

def visualize_tree_from_newick(newick_file: str, 
                              output_file: str = None,
                              show_plot: bool = False) -> None:
    """
    Visualize phylogenetic tree from Newick format
    
    Parameters:
    -----------
    newick_file : str
        Path to Newick file
    output_file : str
        Path to save output image
    show_plot : bool
        Whether to display the plot
    """
    try:
        # Read tree from file
        tree = Phylo.read(newick_file, "newick")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw tree
        Phylo.draw(tree, axes=ax, do_show=False)
        
        # Add title
        ax.set_title(f"Phylogenetic Tree: {os.path.basename(newick_file)}")
        
        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logging.info(f"Tree visualization saved to {output_file}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logging.error(f"Failed to visualize tree: {str(e)}")

def compare_trees(tree_files: List[str], 
                 labels: List[str] = None,
                 output_file: str = "tree_comparison.png") -> None:
    """
    Compare multiple phylogenetic trees side by side
    
    Parameters:
    -----------
    tree_files : List[str]
        List of paths to Newick files
    labels : List[str]
        List of labels for each tree
    output_file : str
        Path to save output image
    """
    if not tree_files:
        logging.warning("No tree files provided for comparison")
        return
        
    n_trees = len(tree_files)
    if labels is None:
        labels = [os.path.basename(f) for f in tree_files]
    
    # Create figure
    fig, axes = plt.subplots(1, n_trees, figsize=(n_trees*5, 6))
    if n_trees == 1:
        axes = [axes]
    
    # Load and draw each tree
    for i, (tree_file, label) in enumerate(zip(tree_files, labels)):
        try:
            tree = Phylo.read(tree_file, "newick")
            Phylo.draw(tree, axes=axes[i], do_show=False)
            axes[i].set_title(label)
        except Exception as e:
            logging.error(f"Failed to load tree {tree_file}: {str(e)}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Tree comparison saved to {output_file}")

def plot_branch_length_distributions(df: pd.DataFrame, 
                                    by_scenario: bool = True,
                                    output_file: str = "branch_lengths.png") -> None:
    """
    Plot distributions of internal branch lengths
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    by_scenario : bool
        Whether to separate distributions by scenario
    output_file : str
        Path to save output image
    """
    # Check if branch length data exists
    if 'mean_internal_branch' not in df.columns:
        logging.warning("Branch length data missing for distribution plot")
        return
        
    # Filter out missing values
    branch_data = df.dropna(subset=['mean_internal_branch'])
    if len(branch_data) == 0:
        logging.warning("No valid branch length data available")
        return
    
    plt.figure(figsize=(12, 8))
    
    if by_scenario and 'scenario' in df.columns:
        # Plot distributions by scenario
        for scenario, color in SCENARIOS_COLORS.items():
            scenario_data = branch_data[branch_data['scenario'] == scenario]
            if len(scenario_data) > 0:
                sns.kdeplot(
                    scenario_data['mean_internal_branch'],
                    label=scenario,
                    color=color,
                    fill=True,
                    alpha=0.3
                )
    else:
        # Plot overall distribution
        sns.histplot(
            branch_data['mean_internal_branch'],
            kde=True,
            bins=30,
            color='darkblue'
        )
    
    plt.xlabel('Mean Internal Branch Length')
    plt.ylabel('Density')
    plt.title('Distribution of Internal Branch Lengths')
    if by_scenario:
        plt.legend(title='Scenario')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logging.info(f"Branch length distribution plot saved to {output_file}")

def create_roc_curves(df: pd.DataFrame, 
                     output_file: str = "roc_curves.png") -> Dict:
    """
    Create ROC curves for introgression detection
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe with introgression information
    output_file : str
        Path to save output image
        
    Returns:
    --------
    Dict
        AUC values for each metric
    """
    # Check required columns
    if not all(col in df.columns for col in ['scenario', 'd_stat']):
        logging.warning("Required columns missing for ROC curve generation")
        return {}
    
    # Prepare binary labels for introgression vs. ILS
    # True = introgression, False = ILS or other scenarios
    has_introgression = ~df['scenario'].isin(['continuous_migration'])
    
    # Metrics to evaluate
    metrics = ['d_stat', 'fst', 'topology_concordance']
    available_metrics = [m for m in metrics if m in df.columns]
    
    plt.figure(figsize=(10, 8))
    auc_values = {}
    
    for metric in available_metrics:
        # Some metrics may be inverse indicators (higher value = less introgression)
        is_inverse = metric in ['topology_concordance']
        
        # Drop rows with NaN values for this metric
        valid_data = df.dropna(subset=[metric])
        if len(valid_data) == 0:
            continue
            
        metric_values = valid_data[metric]
        true_labels = has_introgression[valid_data.index].astype(int)
        
        # For inverse metrics, multiply by -1
        if is_inverse:
            metric_values = -metric_values
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, metric_values)
        roc_auc = auc(fpr, tpr)
        auc_values[metric] = roc_auc
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, 
                label=f'{metric.replace("_", " ").title()} (AUC = {roc_auc:.2f})')
    
    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Introgression Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logging.info(f"ROC curves saved to {output_file}")
    
    return auc_values

def create_correlation_heatmap(df: pd.DataFrame, 
                             output_file: str = "correlation_heatmap.png") -> None:
    """
    Create heatmap of correlations between parameters and metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_file : str
        Path to save output image
    """
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns with too many NaN values
    valid_cols = [col for col in numeric_cols 
                 if df[col].notna().sum() > len(df) * 0.5]
    
    # Calculate correlations
    corr_matrix = df[valid_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .7},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )
    
    plt.title('Correlation Between Parameters and Metrics')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logging.info(f"Correlation heatmap saved to {output_file}")

def create_interactive_dashboard(df: pd.DataFrame, 
                               output_file: str = "dashboard.html") -> None:
    """
    Create an interactive dashboard with multiple visualizations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_file : str
        Path to save output HTML file
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "histogram"}]],
        subplot_titles=("3D Parameter Space", "FST vs D-statistic", 
                       "Branch Lengths by Scenario", "Distribution of D-statistics")
    )
    
    # 1. 3D scatter plot (Parameter Space)
    if all(col in df.columns for col in ['recombination_rate', 'introgression_time', 'fst', 'd_stat']):
        fig.add_trace(
            go.Scatter3d(
                x=df['recombination_rate'],
                y=df['introgression_time'],
                z=df['fst'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df['d_stat'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"Scenario: {s}<br>D-stat: {d:.2f}<br>FST: {f:.2f}" 
                      for s, d, f in zip(df['scenario'], df['d_stat'], df['fst'])],
                hoverinfo='text',
                name="Simulations"
            ),
            row=1, col=1
        )
    
    # 2. Scatter plot (FST vs D-statistic)
    if all(col in df.columns for col in ['fst', 'd_stat', 'scenario']):
        for scenario, color in SCENARIOS_COLORS.items():
            scenario_data = df[df['scenario'] == scenario]
            if len(scenario_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=scenario_data['fst'],
                        y=scenario_data['d_stat'],
                        mode='markers',
                        marker=dict(color=color, size=8),
                        name=scenario
                    ),
                    row=1, col=2
                )
    
    # 3. Box plot (Branch Lengths by Scenario)
    if all(col in df.columns for col in ['mean_internal_branch', 'scenario']):
        for scenario, color in SCENARIOS_COLORS.items():
            scenario_data = df[df['scenario'] == scenario]
            if len(scenario_data) > 0 and not scenario_data['mean_internal_branch'].isna().all():
                fig.add_trace(
                    go.Box(
                        y=scenario_data['mean_internal_branch'].dropna(),
                        name=scenario,
                        marker_color=color
                    ),
                    row=2, col=1
                )
    
    # 4. Histogram (Distribution of D-statistics)
    if 'd_stat' in df.columns:
        for scenario, color in SCENARIOS_COLORS.items():
            scenario_data = df[df['scenario'] == scenario]
            if len(scenario_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=scenario_data['d_stat'],
                        name=scenario,
                        marker_color=color,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
    
    # Update layout
    fig.update_layout(
        title_text="Introgression vs ILS Analysis Dashboard",
        height=900,
        width=1200,
        showlegend=True,
        barmode='overlay'
    )
    
    # Update 3D axis properties
    fig.update_scenes(
        xaxis_title='Recombination Rate',
        yaxis_title='Introgression Time',
        zaxis_title='FST',
        xaxis_type="log"
    )
    
    # Update 2D axis properties
    fig.update_xaxes(title_text="FST", row=1, col=2)
    fig.update_yaxes(title_text="D-statistic", row=1, col=2)
    
    fig.update_xaxes(title_text="Scenario", row=2, col=1)
    fig.update_yaxes(title_text="Internal Branch Length", row=2, col=1)
    
    fig.update_xaxes(title_text="D-statistic", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Save as HTML
    fig.write_html(output_file)
    logging.info(f"Interactive dashboard saved to {output_file}")

def visualize_all(df: pd.DataFrame, output_dir: str = "visualizations") -> None:
    """
    Generate all visualizations for a dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    plot_parameter_space(df, f"{output_dir}/parameter_space.png")
    plot_3d_parameter_space(df, f"{output_dir}/parameter_space_3d.html")
    plot_branch_length_distributions(df, True, f"{output_dir}/branch_lengths_by_scenario.png")
    plot_branch_length_distributions(df, False, f"{output_dir}/branch_lengths_overall.png")
    create_roc_curves(df, f"{output_dir}/roc_curves.png")
    create_correlation_heatmap(df, f"{output_dir}/correlation_heatmap.png")
    create_interactive_dashboard(df, f"{output_dir}/dashboard.html")
    
    logging.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualization tools for ILS vs Introgression")
    parser.add_argument("--data", required=True, help="Path to results CSV file")
    parser.add_argument("--output", default="visualizations", help="Output directory")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    visualize_all(df, args.output)
#!/usr/bin/env python3
"""
Branch Length Visualization
This script creates detailed visualizations of branch length patterns
to help distinguish between introgression and ILS.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from Bio import Phylo
from io import StringIO
import re
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("branch_visualization.log"),
        logging.StreamHandler()
    ]
)

def visualize_branch_comparison(df, output_dir):
    """
    Create detailed visualizations comparing branch length distributions
    between introgression and ILS scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing simulation results.
    output_dir : str
        Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If scenario or has_introgression isn't present, do some default
    # classification or fallback. E.g., scenario may be 'continuous_migration' for ILS, 
    # or might not be present at all.
    if 'scenario' not in df.columns:
        logging.warning("Column 'scenario' not found. Assigning default scenario='ILS'.")
        df['scenario'] = 'ILS'
    
    if 'has_introgression' not in df.columns:
        # Mark introgression present if scenario is not 'continuous_migration'
        # or if scenario name suggests gene flow. Adjust this logic as needed.
        df['has_introgression'] = ~df['scenario'].isin(['continuous_migration', 'ILS'])
    
    # Filter out rows with missing branch data
    if not {'mean_internal_branch', 'std_internal_branch'}.issubset(df.columns):
        logging.error("The dataframe does not contain 'mean_internal_branch' or 'std_internal_branch' columns.")
        return
    
    df_clean = df.dropna(subset=['mean_internal_branch', 'std_internal_branch'])
    
    if len(df_clean) < 10:
        logging.error("Not enough data points with branch length information.")
        return
    
    # 1. Violin plot comparing distributions
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x='has_introgression',
        y='mean_internal_branch',
        data=df_clean,
        inner='quartile',
        palette=['#3498db', '#e74c3c']  # Blue for False (ILS), Red for True (Introgression)
    )
    
    plt.xlabel('Introgression Present')
    plt.ylabel('Mean Internal Branch Length')
    plt.title('Distribution of Internal Branch Lengths: Introgression vs ILS')
    
    # Add statistical test (independent t-test)
    try:
        ils_mean_branch = df_clean[~df_clean['has_introgression']]['mean_internal_branch']
        intro_mean_branch = df_clean[df_clean['has_introgression']]['mean_internal_branch']
        
        if len(ils_mean_branch) >= 2 and len(intro_mean_branch) >= 2:
            stats_result = stats.ttest_ind(ils_mean_branch, intro_mean_branch, equal_var=False)
            plt.annotate(
                f"t-test p-value: {stats_result.pvalue:.4f}",
                xy=(0.5, 0.02),
                xycoords='figure fraction',
                ha='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        else:
            logging.warning("Not enough data to perform t-test on mean branch length.")
    except Exception as e:
        logging.error(f"Error while performing t-test: {str(e)}")
    
    plt.savefig(os.path.join(output_dir, "branch_length_violin.png"), dpi=300)
    plt.close()
    
    # 2. Branch length vs branch length variability scatterplot
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    
    # Convert True/False to int for coloring
    color_vals = df_clean['has_introgression'].astype(int)
    
    scatter = plt.scatter(
        df_clean['mean_internal_branch'],
        df_clean['std_internal_branch'],
        c=color_vals,
        cmap='coolwarm',
        alpha=0.7,
        s=50
    )
    
    plt.xlabel('Mean Internal Branch Length')
    plt.ylabel('Standard Deviation of Branch Lengths')
    plt.title('Branch Length vs. Branch Length Variability')
    plt.grid(alpha=0.3)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
               markersize=10, label='ILS (has_introgression=False)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
               markersize=10, label='Introgression (has_introgression=True)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Second subplot: Kernel density estimation
    plt.subplot(2, 1, 2)
    
    # Separate the data by introgression status
    ils_data = df_clean[~df_clean['has_introgression']]
    intro_data = df_clean[df_clean['has_introgression']]
    
    # Create KDE plot for ILS
    if len(ils_data) > 5:
        sns.kdeplot(
            x=ils_data['mean_internal_branch'],
            y=ils_data['std_internal_branch'],
            fill=True,
            alpha=0.5,
            color='#3498db',
            label='ILS'
        )
    
    # Create KDE plot for introgression
    if len(intro_data) > 5:
        sns.kdeplot(
            x=intro_data['mean_internal_branch'],
            y=intro_data['std_internal_branch'],
            fill=True,
            alpha=0.5,
            color='#e74c3c',
            label='Introgression'
        )
    
    plt.xlabel('Mean Internal Branch Length')
    plt.ylabel('Standard Deviation of Branch Lengths')
    plt.title('Density Distribution of Branch Length Characteristics')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branch_length_scatter.png"), dpi=300)
    plt.close()
    
    # 3. Branch length ratio analysis (if columns exist)
    if {'min_internal_branch', 'max_internal_branch'}.issubset(df_clean.columns):
        df_clean['branch_ratio'] = df_clean['max_internal_branch'] / df_clean['min_internal_branch']
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(
            x='has_introgression',
            y='branch_ratio',
            data=df_clean,
            inner='quartile',
            palette=['#3498db', '#e74c3c']
        )
        
        plt.xlabel('Introgression Present')
        plt.ylabel('Branch Length Ratio (Max/Min)')
        plt.title('Branch Length Ratio: Introgression vs ILS')
        
        # Add statistical test
        try:
            ils_ratio = df_clean[~df_clean['has_introgression']]['branch_ratio']
            intro_ratio = df_clean[df_clean['has_introgression']]['branch_ratio']
            if len(ils_ratio) >= 2 and len(intro_ratio) >= 2:
                stats_result = stats.ttest_ind(ils_ratio, intro_ratio, equal_var=False)
                plt.annotate(
                    f"t-test p-value: {stats_result.pvalue:.4f}",
                    xy=(0.5, 0.02),
                    xycoords='figure fraction',
                    ha='center',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        except Exception as e:
            logging.error(f"Error while performing t-test on branch ratio: {str(e)}")
        
        plt.savefig(os.path.join(output_dir, "branch_ratio_violin.png"), dpi=300)
        plt.close()
        
        # 4. ROC curve for branch ratio as a diagnostic
        plt.figure(figsize=(10, 8))
        try:
            fpr, tpr, _ = roc_curve(df_clean['has_introgression'], df_clean['branch_ratio'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'Branch Ratio (AUC = {roc_auc:.2f})')
            # Reference line
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve: Branch Length Ratio for Detecting Introgression')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, "branch_ratio_roc.png"), dpi=300)
            plt.close()
        except ValueError as e:
            logging.error(f"Error computing ROC curve: {str(e)}")

def analyze_tree_files(tree_dir, output_dir):
    """
    Analyze and visualize Newick tree files

    Parameters
    ----------
    tree_dir : str
        Directory containing Newick tree files.
    output_dir : str
        Directory to save visualizations.
    """
    if not os.path.exists(tree_dir):
        logging.error(f"Tree directory not found: {tree_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Newick files
    newick_files = []
    for root, _, files in os.walk(tree_dir):
        for file in files:
            if file.endswith('.nwk'):
                newick_files.append(os.path.join(root, file))
    
    if not newick_files:
        logging.warning(f"No Newick files found in {tree_dir}")
        return
    
    logging.info(f"Found {len(newick_files)} Newick files in {tree_dir}")
    
    # Group files by scenario
    scenario_pattern = r'(base|rapid_radiation|bottleneck|continuous_migration)'
    scenario_trees = {}
    
    for file_path in newick_files:
        # Try to extract scenario from filename
        match = re.search(scenario_pattern, os.path.basename(file_path))
        if match:
            scenario = match.group(1)
        else:
            # Default scenario if no match
            scenario = "other"
        
        if scenario not in scenario_trees:
            scenario_trees[scenario] = []
        scenario_trees[scenario].append(file_path)
    
    # Analyze a sample of trees from each scenario
    sample_size = 5  # Number of trees to analyze per scenario
    
    for scenario, tree_files in scenario_trees.items():
        if not tree_files:
            continue
        
        # Sample trees if there are many
        if len(tree_files) > sample_size:
            import random
            sampled_trees = random.sample(tree_files, sample_size)
        else:
            sampled_trees = tree_files
        
        # Create visualization directory for this scenario
        scenario_dir = os.path.join(output_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Process each tree
        for i, tree_file in enumerate(sampled_trees):
            try:
                with open(tree_file, 'r') as f:
                    newick_str = f.read().strip()
                
                # Parse tree
                tree = Phylo.read(StringIO(newick_str), 'newick')
                
                # Draw tree
                plt.figure(figsize=(12, 8))
                Phylo.draw(tree, do_show=False)
                plt.title(f"{scenario} - {os.path.basename(tree_file)}")
                plt.savefig(os.path.join(scenario_dir, f"tree_{i+1}.png"), dpi=300)
                plt.close()
                
                # Extract and save branch lengths
                branch_lengths = []
                for clade in tree.find_clades():
                    if clade.branch_length is not None:
                        branch_lengths.append(clade.branch_length)
                
                # Plot branch length distribution
                if branch_lengths:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(branch_lengths, kde=True, bins=20)
                    plt.xlabel('Branch Length')
                    plt.ylabel('Frequency')
                    plt.title(f'Branch Length Distribution - {scenario} - {os.path.basename(tree_file)}')
                    plt.savefig(os.path.join(scenario_dir, f"branch_dist_{i+1}.png"), dpi=300)
                    plt.close()
                    
            except Exception as e:
                logging.error(f"Error processing {tree_file}: {str(e)}")
    
    # Create composite comparison of branch length distributions across scenarios
    plt.figure(figsize=(12, 8))
    
    # Process one representative tree from each scenario
    legend_elements = []
    for scenario, tree_files in scenario_trees.items():
        if tree_files:
            # Take the first tree file for a quick comparison
            first_tree_path = tree_files[0]
            try:
                with open(first_tree_path, 'r') as f:
                    newick_str = f.read().strip()
                tree = Phylo.read(StringIO(newick_str), 'newick')
                
                branch_lengths = []
                for clade in tree.find_clades():
                    if clade.branch_length is not None:
                        branch_lengths.append(clade.branch_length)
                
                if branch_lengths:
                    sns.kdeplot(branch_lengths, label=scenario)
                    
                    # Add to legend
                    legend_elements.append(
                        Line2D([0], [0], color=plt.gca().lines[-1].get_color(), 
                               lw=2, label=scenario)
                    )
            except Exception as e:
                logging.error(f"Error processing sample tree for {scenario}: {str(e)}")
    
    plt.xlabel('Branch Length')
    plt.ylabel('Density')
    plt.title('Branch Length Distributions by Scenario')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "scenario_branch_comparison.png"), dpi=300)
    plt.close()
    
    logging.info(f"Tree analysis completed. Results saved to {output_dir}")

def create_conceptual_diagrams(output_dir):
    """
    Create conceptual diagrams illustrating branch length patterns
    in introgression vs ILS scenarios.

    Parameters
    ----------
    output_dir : str
        Directory to save diagrams.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Conceptual ILS diagram
    plt.figure(figsize=(10, 8))
    
    # Define species tree timescale
    plt.plot([0, 0], [0, 3], 'k-', linewidth=2)  # Species A
    plt.plot([1, 1], [0, 3], 'k-', linewidth=2)  # Species B
    plt.plot([2, 2], [0, 3], 'k-', linewidth=2)  # Species C (outgroup)
    
    # Connect species (species tree)
    plt.plot([0, 1], [3, 3], 'k-', linewidth=2)
    plt.plot([0.5, 2], [3.5, 3.5], 'k-', linewidth=2)
    plt.plot([0.5, 0.5], [3, 3.5], 'k-', linewidth=2)
    
    # Add species labels
    plt.text(0, -0.2, "Species A", ha='center')
    plt.text(1, -0.2, "Species B", ha='center')
    plt.text(2, -0.2, "Species C", ha='center')
    
    # Draw gene tree (ILS - discordant)
    plt.plot([0, 0.7], [1, 2.5], 'b-', linewidth=1.5, alpha=0.7)  
    plt.plot([1, 0.7], [1, 2.5], 'b-', linewidth=1.5, alpha=0.7)  
    plt.plot([0.7, 1.5], [2.5, 3.2], 'b-', linewidth=1.5, alpha=0.7)
    plt.plot([2, 1.5], [1, 3.2], 'b-', linewidth=1.5, alpha=0.7)  
    
    # Add gene tree labels
    plt.text(0, 1, "Gene A", color='blue', ha='right')
    plt.text(1, 1, "Gene B", color='blue', ha='left')
    plt.text(2, 1, "Gene C", color='blue', ha='left')
    
    # Add time annotations
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.6)
    plt.text(2.5, 3, "Species A/B Split", va='center')
    
    plt.axhline(y=3.5, color='gray', linestyle='--', alpha=0.6)
    plt.text(2.5, 3.5, "Split from Outgroup", va='center')
    
    # Add ILS explanation
    plt.text(1, 4, "Incomplete Lineage Sorting (ILS)", ha='center', 
             fontsize=14, fontweight='bold')
    plt.text(1, 4.3, "Gene tree differs from species tree due to ancestral polymorphism", 
             ha='center')
    plt.text(1, 4.6, "Branch lengths can follow an exponential distribution", ha='center')
    
    plt.xlim(-0.5, 3)
    plt.ylim(-0.5, 5)
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, "ils_conceptual.png"), dpi=300)
    plt.close()
    
    # 2. Conceptual introgression diagram
    plt.figure(figsize=(10, 8))
    
    # Define species tree timescale
    plt.plot([0, 0], [0, 3], 'k-', linewidth=2)  # Species A
    plt.plot([1, 1], [0, 3], 'k-', linewidth=2)  # Species B
    plt.plot([2, 2], [0, 3], 'k-', linewidth=2)  # Species C (outgroup)
    
    # Connect species (species tree)
    plt.plot([0, 1], [3, 3], 'k-', linewidth=2)
    plt.plot([0.5, 2], [3.5, 3.5], 'k-', linewidth=2)
    plt.plot([0.5, 0.5], [3, 3.5], 'k-', linewidth=2)
    
    # Add species labels
    plt.text(0, -0.2, "Species A", ha='center')
    plt.text(1, -0.2, "Species B", ha='center')
    plt.text(2, -0.2, "Species C", ha='center')
    
    # Draw introgression arrow
    plt.arrow(0, 1.5, 0.9, 0, head_width=0.1, head_length=0.1, 
              fc='red', ec='red', linewidth=1.5, length_includes_head=True)
    plt.text(0.45, 1.7, "Introgression", color='red')
    
    # Draw gene tree (introgression-based discordance)
    plt.plot([0, 0.5], [1, 1.5], 'b-', linewidth=1.5, alpha=0.7)  
    plt.plot([1, 0.5], [1, 1.5], 'b-', linewidth=1.5, alpha=0.7)  
    plt.plot([0.5, 1.5], [1.5, 3.2], 'b-', linewidth=1.5, alpha=0.7)
    plt.plot([2, 1.5], [1, 3.2], 'b-', linewidth=1.5, alpha=0.7)  
    
    # Add gene tree labels
    plt.text(0, 1, "Gene A", color='blue', ha='right')
    plt.text(1, 1, "Gene B", color='blue', ha='left')
    plt.text(2, 1, "Gene C", color='blue', ha='left')
    
    # Add time annotations
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.6)
    plt.text(2.5, 3, "Species A/B Split", va='center')
    plt.axhline(y=3.5, color='gray', linestyle='--', alpha=0.6)
    plt.text(2.5, 3.5, "Split from Outgroup", va='center')
    
    plt.axhline(y=1.5, color='red', linestyle='--', alpha=0.6)
    plt.text(2.5, 1.5, "Introgression Event", va='center', color='red')
    
    # Add introgression explanation
    plt.text(1, 4, "Introgression", ha='center', fontsize=14, fontweight='bold', color='red')
    plt.text(1, 4.3, "Gene B more closely related to Gene A due to gene flow", ha='center')
    plt.text(1, 4.6, "Branch lengths can show bimodal distribution", ha='center')
    
    plt.xlim(-0.5, 3)
    plt.ylim(-0.5, 5)
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, "introgression_conceptual.png"), dpi=300)
    plt.close()
    
    # 3. Theoretical branch length distribution comparison
    plt.figure(figsize=(12, 8))
    
    x = np.linspace(0, 5, 1000)
    
    # ILS ~ exponential distribution
    ils_dist = stats.expon.pdf(x, scale=1)
    plt.plot(x, ils_dist, 'b-', linewidth=2, label='ILS (Exponential)')
    
    # Introgression ~ bimodal distribution (example)
    introgression_dist1 = stats.norm.pdf(x, loc=1, scale=0.3)
    introgression_dist2 = stats.norm.pdf(x, loc=3, scale=0.5)
    introgression_dist = 0.6 * introgression_dist1 + 0.4 * introgression_dist2
    plt.plot(x, introgression_dist, 'r-', linewidth=2, label='Introgression (Bimodal)')
    
    plt.xlabel('Internal Branch Length')
    plt.ylabel('Density')
    plt.title('Theoretical Branch Length Distributions: ILS vs Introgression')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.figtext(
        0.5, 0.01,
        ("ILS typically results in exponentially distributed branch lengths,\n"
         "while introgression may show bimodality with a second peak corresponding to the introgression time."),
        ha='center', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust for the text at the bottom
    plt.savefig(os.path.join(output_dir, "theoretical_distributions.png"), dpi=300)
    plt.close()
    
    logging.info(f"Conceptual diagrams created and saved to {output_dir}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Create branch length visualizations")
    parser.add_argument("--results", help="Path to simulation results CSV file")
    parser.add_argument("--tree_dir", help="Directory containing Newick tree files")
    parser.add_argument("--output_dir", default="branch_visualizations", help="Output directory")
    parser.add_argument("--conceptual", action="store_true", help="Create conceptual diagrams")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process results CSV if provided
    if args.results and os.path.exists(args.results):
        logging.info(f"Loading simulation results from {args.results}")
        df = pd.read_csv(args.results)
        visualize_branch_comparison(df, args.output_dir)
    else:
        if args.results:
            logging.warning(f"Results file not found or not specified: {args.results}")
    
    # Process tree files if provided
    if args.tree_dir and os.path.exists(args.tree_dir):
        trees_output_dir = os.path.join(args.output_dir, "tree_analysis")
        analyze_tree_files(args.tree_dir, trees_output_dir)
    else:
        if args.tree_dir:
            logging.warning(f"Tree directory not found or not specified: {args.tree_dir}")
    
    # Create conceptual diagrams if requested
    if args.conceptual:
        conceptual_dir = os.path.join(args.output_dir, "conceptual_diagrams")
        create_conceptual_diagrams(conceptual_dir)
    
    logging.info(f"Branch visualizations completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

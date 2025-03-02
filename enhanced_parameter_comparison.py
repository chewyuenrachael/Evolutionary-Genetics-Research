import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import glob
import re

def find_results_files(results_dir):
    """
    Find all relevant results files using multiple strategies
    
    Parameters:
    -----------
    results_dir : str
        Directory to search for results
        
    Returns:
    --------
    list
        List of file paths
    """
    # Strategy 1: Look for CSV files with 'sweep' or 'parameter' in the name
    files = []
    for root, _, filenames in os.walk(results_dir):
        for filename in filenames:
            if filename.endswith('.csv') and ('sweep' in filename.lower() or 'parameter' in filename.lower()):
                files.append(os.path.join(root, filename))
    
    # Strategy 2: Look for any results.csv file
    if not files:
        for root, _, filenames in os.walk(results_dir):
            for filename in filenames:
                if filename.endswith('_results.csv') or filename == 'results.csv':
                    files.append(os.path.join(root, filename))
    
    # Strategy 3: Look for any CSV file if we still don't have results
    if not files:
        for root, _, filenames in os.walk(results_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    files.append(os.path.join(root, filename))
    
    # Print debug info
    print(f"Found {len(files)} potential results files")
    for file in files[:10]:  # Print first 10 files
        print(f"  - {file}")
    if len(files) > 10:
        print(f"  - ... and {len(files) - 10} more")
    
    return files

def examine_file_content(file_path):
    """
    Print information about the columns in a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\nExamining file: {file_path}")
        print(f"Shape: {df.shape}")
        print("Columns:")
        for col in df.columns:
            # Get unique values (limited to 5)
            unique_vals = df[col].dropna().unique()[:5]
            print(f"  - {col}: {len(df[col].dropna())} non-null values, {len(df[col].unique())} unique values")
            print(f"    Example values: {unique_vals}")
        
        # Check for key columns we need
        key_columns = ['scenario', 'mean_internal_branch', 'std_internal_branch', 'd_stat', 'fst']
        present_keys = [col for col in key_columns if col in df.columns]
        print(f"Key columns present: {present_keys}")
        
        # Check for parameter columns
        param_columns = ['introgression_proportion', 'introgression_time', 'recombination_rate']
        present_params = [col for col in param_columns if col in df.columns]
        print(f"Parameter columns present: {present_params}")
        
        # Identify alternative parameter columns
        alt_params = []
        for param in param_columns:
            base_param = param.split('_')[0]
            matches = [col for col in df.columns if base_param in col.lower() and col not in param_columns]
            if matches:
                alt_params.extend(matches)
        
        if alt_params:
            print(f"Potential alternative parameter columns: {alt_params}")
        
        return df.columns.tolist()
    except Exception as e:
        print(f"Error examining {file_path}: {str(e)}")
        return []

def compare_parameter_effects(results_dir, output_dir="parameter_visualizations"):
    """
    Create comprehensive visualizations comparing how different parameters
    affect branch length distributions across evolutionary scenarios.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing simulation results CSV files
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find results files
    results_files = find_results_files(results_dir)
    
    if not results_files:
        print("No results files found. Please check the directory path.")
        return
    
    # Examine a sample of files to understand structure
    print("\nExamining sample files to understand structure:")
    all_columns = []
    for file in results_files[:3]:  # Check first 3 files
        columns = examine_file_content(file)
        all_columns.extend(columns)
    
    # Get unique columns
    all_columns = list(set(all_columns))
    print(f"\nTotal unique columns across files: {len(all_columns)}")
    
    # Load and combine results
    dfs = []
    for file in results_files:
        try:
            df = pd.read_csv(file)
            # Add file identifier
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
            print(f"Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        print("No valid results files could be loaded")
        return
    
    # Look for common columns across all dataframes
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    print(f"Common columns across all files: {common_columns}")
    
    # If there are few common columns, try to find equivalent columns
    if len(common_columns) < 5:
        print("Few common columns found. Attempting to align column names...")
        # Create a mapping of column names
        column_mapping = {}
        target_cols = ['scenario', 'mean_internal_branch', 'std_internal_branch', 'd_stat', 'fst']
        
        for target in target_cols:
            found = False
            # Check if column exists as is
            if all(target in df.columns for df in dfs):
                column_mapping[target] = target
                found = True
            else:
                # Try to find equivalent
                base_name = target.split('_')[0]
                for df in dfs:
                    matches = [col for col in df.columns if base_name in col.lower()]
                    if matches:
                        column_mapping[target] = matches[0]
                        found = True
                        break
            
            if found:
                print(f"Mapped {target} -> {column_mapping.get(target, 'Not found')}")
        
        # Rename columns in each dataframe
        for i, df in enumerate(dfs):
            rename_dict = {}
            for target, mapped in column_mapping.items():
                if mapped in df.columns and mapped != target:
                    rename_dict[mapped] = target
            
            if rename_dict:
                dfs[i] = df.rename(columns=rename_dict)
    
    # Combine all data
    try:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataframe: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    except Exception as e:
        print(f"Error combining dataframes: {str(e)}")
        return
    
    # Check for essential columns
    essential_cols = ['scenario', 'mean_internal_branch']
    if not all(col in combined_df.columns for col in essential_cols):
        print(f"Missing essential columns. We need at least {essential_cols}")
        print(f"Available columns: {combined_df.columns.tolist()}")
        return
    
    # Save the combined data for reference
    combined_df.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)
    print(f"Saved combined results to {os.path.join(output_dir, 'combined_results.csv')}")
    
    # Plot branch length distribution by scenario (this should work even with limited data)
    plot_branch_length_by_scenario(combined_df, output_dir)
    
    # Check if we have parameter columns for detailed analysis
    parameter_cols = ['introgression_proportion', 'introgression_time', 'recombination_rate']
    available_params = [col for col in parameter_cols if col in combined_df.columns]
    
    if available_params:
        print(f"Found parameter columns: {available_params}")
        for param in available_params:
            plot_parameter_effect(combined_df, param, output_dir)
    else:
        print("No parameter columns found for detailed analysis")
        # Try to identify alternative parameter columns
        alt_params = []
        for param in parameter_cols:
            base_param = param.split('_')[0]
            matches = [col for col in combined_df.columns if base_param in col.lower()]
            if matches:
                alt_params.extend(matches)
        
        if alt_params:
            print(f"Found alternative parameter columns: {alt_params}")
            for param in alt_params:
                plot_parameter_effect(combined_df, param, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

def plot_parameter_effect(df, param_col, output_dir):
    """
    Plot how a parameter affects branch length metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined results dataframe
    param_col : str
        Parameter column name
    output_dir : str
        Directory to save visualizations
    """
    # Select relevant metrics
    metrics = ['mean_internal_branch', 'std_internal_branch', 'd_stat', 'fst']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"No relevant metrics found for {param_col} analysis")
        return
    
    # Create plot
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4*len(available_metrics)), sharex=True)
    
    # If only one metric, axes is not a list
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        # Get data for this metric
        plot_df = df[[param_col, metric, 'scenario']].dropna()
        
        if len(plot_df) < 5:
            print(f"Not enough data for {metric} analysis")
            continue
        
        # Create scatter plot with regression lines for each scenario
        sns.scatterplot(
            data=plot_df,
            x=param_col,
            y=metric,
            hue='scenario',
            palette='viridis',
            alpha=0.7,
            ax=axes[i]
        )
        
        # Add regression line for overall trend
        try:
            sns.regplot(
                data=plot_df,
                x=param_col,
                y=metric,
                scatter=False,
                ax=axes[i],
                color='black',
                line_kws={'linestyle': '--'}
            )
        except Exception as e:
            print(f"Error adding regression line: {str(e)}")
        
        axes[i].set_title(f'Effect of {param_col.replace("_", " ").title()} on {metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        # Use log scale for rate parameters
        if 'rate' in param_col.lower():
            try:
                axes[i].set_xscale('log')
            except Exception as e:
                print(f"Error setting log scale: {str(e)}")
        
        # Add grid for better readability
        axes[i].grid(alpha=0.3)
    
    param_title = param_col.replace('_', ' ').title()
    axes[-1].set_xlabel(param_title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param_col}_effect.png"), dpi=300)
    plt.close()
    print(f"Saved {param_col} effect plot")

def plot_branch_length_by_scenario(df, output_dir):
    """Create comparative visualization of branch length distributions by scenario"""
    # Check for required columns
    if 'mean_internal_branch' not in df.columns or 'scenario' not in df.columns:
        print("Required columns missing for branch length by scenario visualization")
        return
    
    # Select relevant data
    plot_df = df[['mean_internal_branch', 'scenario']].dropna()
    
    if len(plot_df) < 10:
        print("Not enough data for branch length comparison")
        return
    
    print(f"Plotting branch length distributions for {len(plot_df)} data points across scenarios")
    print(f"Scenarios: {plot_df['scenario'].unique()}")
    
    # Create violin plot
    plt.figure(figsize=(12, 8))
    
    try:
        # Create violin plot with inner box plot
        ax = sns.violinplot(
            data=plot_df,
            x='scenario',
            y='mean_internal_branch',
            palette='viridis',
            inner='box',
            cut=0
        )
        
        # Add individual data points
        sns.stripplot(
            data=plot_df,
            x='scenario',
            y='mean_internal_branch',
            color='black',
            alpha=0.3,
            size=4
        )
    except Exception as e:
        print(f"Error creating violin plot: {str(e)}")
        # Fallback to boxplot
        try:
            ax = sns.boxplot(
                data=plot_df,
                x='scenario',
                y='mean_internal_branch',
                palette='viridis'
            )
        except Exception as e2:
            print(f"Error creating boxplot fallback: {str(e2)}")
            return
    
    # Add labels and title
    plt.title('Branch Length Distribution by Evolutionary Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Mean Internal Branch Length')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branch_length_by_scenario.png"), dpi=300)
    plt.close()
    print("Saved branch length by scenario plot")
    
    # Create density plots for more detailed comparison
    plt.figure(figsize=(12, 8))
    
    try:
        # Create KDE plot for each scenario
        for scenario in plot_df['scenario'].unique():
            scenario_data = plot_df[plot_df['scenario'] == scenario]
            if len(scenario_data) >= 5:
                sns.kdeplot(
                    scenario_data['mean_internal_branch'],
                    label=scenario,
                    fill=True,
                    alpha=0.3
                )
    except Exception as e:
        print(f"Error creating density plot: {str(e)}")
        return
    
    # Add labels and title
    plt.title('Branch Length Density by Evolutionary Scenario')
    plt.xlabel('Mean Internal Branch Length')
    plt.ylabel('Density')
    plt.legend(title='Scenario')
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branch_length_density_by_scenario.png"), dpi=300)
    plt.close()
    print("Saved branch length density plot")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparative parameter visualizations")
    parser.add_argument("--results_dir", required=True, help="Directory containing simulation results")
    parser.add_argument("--output_dir", default="parameter_visualizations", help="Directory to save visualizations")
    parser.add_argument("--examine_file", help="Path to a specific file to examine")
    
    args = parser.parse_args()
    
    if args.examine_file:
        examine_file_content(args.examine_file)
    else:
        compare_parameter_effects(args.results_dir, args.output_dir)
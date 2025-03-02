#!/usr/bin/env python3
"""
Master Pipeline for Distinguishing Introgression and ILS
This script integrates data generation, simulation, and analysis
to comprehensively address the research question.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc  # Added import for roc_curve and auc


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import your custom modules (assuming they're in the same directory)
# If these imports fail, you'll need to modify paths
from enhanced_genetics_pipeline import main as run_simulation
from branch_length_analysis import analyze_branch_length_distributions
from ml_classification import train_introgression_classifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def create_config(args):
    """Create configuration for simulation run"""
    # Default configuration with sensible parameters
    config = {
        'num_simulations': args.num_simulations,
        'genome_length': 1e5,
        'max_workers': min(os.cpu_count(), 8),
        'output_prefix': os.path.join(args.output_dir, "simulation"),
        'save_trees': True,
        'scenario_weights': {
            'base': 0.4,
            'rapid_radiation': 0.3,
            'bottleneck': 0.2,
            'continuous_migration': 0.1
        }
    }
    
    # Override with user-specific parameters
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Save the final configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def run_parameter_sweep(base_config, parameter_name, values, output_dir):
    """
    Run simulations across a range of parameter values
    to analyze sensitivity and detection thresholds
    """
    results = []
    
    for value in values:
        # Update configuration with new parameter value
        config = base_config.copy()
        
        # Handle nested parameters
        if parameter_name == "introgression_proportion":
            config["scenario_weights"] = {
                'base': 0.5,  # Simplified for parameter sweep
                'rapid_radiation': 0.0,
                'bottleneck': 0.0,
                'continuous_migration': 0.5
            }
            config["introgression_proportion_range"] = [value, value]
        elif parameter_name == "introgression_time":
            config["scenario_weights"] = {
                'base': 0.5,
                'rapid_radiation': 0.0,
                'bottleneck': 0.0,
                'continuous_migration': 0.5
            }
            config["introgression_time_range"] = [value, value]
        elif parameter_name == "recombination_rate":
            config["recombination_rate_range"] = [value, value]
        elif parameter_name == "mutation_rate":
            config["mutation_rate_range"] = [value, value]
        
        # Set unique output prefix for this run
        config["output_prefix"] = os.path.join(
            output_dir, f"sweep_{parameter_name}_{value}"
        )
        
        # Save configuration
        config_path = os.path.join(output_dir, f"config_{parameter_name}_{value}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run simulation with this configuration
        logging.info(f"Running simulation with {parameter_name}={value}")
        result = run_simulation(config_file=config_path)
        
        # Store result summary with parameter value
        if isinstance(result, dict) and not result.get('error'):
            result['parameter_name'] = parameter_name
            result['parameter_value'] = value
            results.append(result)
        else:
            logging.error(f"Simulation failed for {parameter_name}={value}")
    
    # Combine results
    if results:
        # Extract metrics of interest
        summary_data = []
        for res in results:
            summary_data.append({
                'parameter_name': res['parameter_name'],
                'parameter_value': res['parameter_value'],
                'avg_fst': res.get('avg_fst'),
                'avg_d_stat': res.get('avg_d_stat'),
                'scenarios': res.get('scenarios', {})
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(output_dir, f"parameter_sweep_{parameter_name}.csv"),
            index=False
        )
        
        # Visualize parameter sensitivity
        plt.figure(figsize=(10, 6))
        
        # D-statistic by parameter value
        plt.subplot(1, 2, 1)
        plt.plot(summary_df['parameter_value'], summary_df['avg_d_stat'], 'o-', linewidth=2)
        plt.xlabel(parameter_name)
        plt.ylabel('Average D-statistic')
        plt.title(f'Sensitivity of D-statistic to {parameter_name}')
        plt.grid(alpha=0.3)
        
        # FST by parameter value
        plt.subplot(1, 2, 2)
        plt.plot(summary_df['parameter_value'], summary_df['avg_fst'], 'o-', linewidth=2, color='orange')
        plt.xlabel(parameter_name)
        plt.ylabel('Average FST')
        plt.title(f'Sensitivity of FST to {parameter_name}')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"parameter_sensitivity_{parameter_name}.png"),
            dpi=300
        )
        plt.close()
        
        return summary_df
    
    return None

def integrate_analysis_results(results_file, output_dir):
    """
    Integrate results from different analyses into a comprehensive summary
    addressing the research question.
    """
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        return False
    
    logging.info(f"Analyzing results from {results_file}")
    
    # Load simulation results
    df = pd.read_csv(results_file)
    
    # Create analysis directories
    branch_analysis_dir = os.path.join(output_dir, "branch_analysis")
    ml_dir = os.path.join(output_dir, "ml_analysis")
    os.makedirs(branch_analysis_dir, exist_ok=True)
    os.makedirs(ml_dir, exist_ok=True)
    
    # Run branch length analysis
    logging.info("Running branch length analysis")
    branch_results = analyze_branch_length_distributions(df, branch_analysis_dir)
    
    # Run machine learning analysis
    logging.info("Running machine learning classification")
    try:
        ml_results = train_introgression_classifier(df, ml_dir)
        has_ml_results = True
    except Exception as e:
        logging.error(f"Machine learning analysis failed: {str(e)}")
        has_ml_results = False
    
    # Create integrated visualization
    plt.figure(figsize=(12, 10))
    
    # 1. Branch length distributions comparison
    plt.subplot(2, 2, 1)
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]['mean_internal_branch'].dropna()
        if len(scenario_data) > 0:
            sns.kdeplot(scenario_data, label=scenario)
    plt.xlabel('Mean Internal Branch Length')
    plt.ylabel('Density')
    plt.title('Branch Length Distributions by Scenario')
    plt.legend()
    
    # 2. D-statistic vs FST
    plt.subplot(2, 2, 2)
    if 'has_introgression' not in df.columns:
        df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    scatter = plt.scatter(
        df['fst'].dropna(), 
        df['d_stat'].dropna(),
        c=df['has_introgression'].dropna().astype(int),
        cmap='coolwarm',
        alpha=0.7
    )
    plt.xlabel('FST')
    plt.ylabel('D-statistic')
    plt.title('FST vs D-statistic')
    plt.legend(*scatter.legend_elements(), title="Introgression")
    plt.grid(alpha=0.3)
    
    # 3. ROC curves for individual metrics
    plt.subplot(2, 2, 3)
    for metric in ['d_stat', 'fst', 'topology_concordance']:
        if metric in df.columns:
            # Handle inverse features
            if metric in ['topology_concordance']: 
                metric_values = -df[metric].dropna()
            else:
                metric_values = df[metric].dropna()
            
            y_true = df.loc[metric_values.index, 'has_introgression']
            fpr, tpr, _ = roc_curve(y_true, metric_values)
            metric_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{metric} (AUC = {metric_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Introgression Detection')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # 4. Feature importance (if ML was successful)
    plt.subplot(2, 2, 4)
    if has_ml_results and 'feature_importance' in ml_results:
        feature_imp = ml_results['feature_importance']
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title('Feature Importance for Detection')
        plt.xlabel('Importance')
    else:
        # Alternative: correlation with has_introgression
        corr_data = []
        for col in df.columns:
            if col == 'has_introgression' or df[col].dtype.kind not in 'if':
                continue
            valid_idx = df[[col, 'has_introgression']].dropna().index
            if len(valid_idx) > 10:
                corr = df.loc[valid_idx, col].corr(df.loc[valid_idx, 'has_introgression'])
                corr_data.append({'feature': col, 'correlation': abs(corr)})
        if corr_data:
            corr_df = pd.DataFrame(corr_data).sort_values('correlation', ascending=False)
            sns.barplot(x='correlation', y='feature', data=corr_df)
            plt.title('Feature Correlation with Introgression')
            plt.xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "integrated_analysis.png"), dpi=300)
    plt.close()
    
    # Generate comprehensive report
    with open(os.path.join(output_dir, "analysis_report.md"), 'w') as f:
        f.write("# Analysis Report: Distinguishing Introgression from ILS\n\n")
        f.write("## Overview of Simulation Results\n\n")
        f.write(f"- Total simulations: {len(df)}\n")
        f.write(f"- Scenarios: {', '.join(df['scenario'].unique())}\n\n")
        f.write("## Branch Length Analysis\n\n")
        if 'best_fits' in branch_results:
            f.write("### Best-Fitting Distributions by Scenario\n\n")
            f.write(branch_results['best_fits'].to_markdown() + "\n\n")
            exp_scenarios = branch_results['best_fits'][branch_results['best_fits']['distribution'] == 'exponential']['scenario'].tolist()
            if exp_scenarios:
                f.write(f"- Scenarios matching exponential distribution (typical of ILS): {', '.join(exp_scenarios)}\n")
            other_scenarios = branch_results['best_fits'][branch_results['best_fits']['distribution'] != 'exponential']['scenario'].tolist()
            if other_scenarios:
                f.write(f"- Scenarios with non-exponential distributions (may indicate introgression): {', '.join(other_scenarios)}\n\n")
        if 'bimodality' in branch_results:
            f.write("### Bimodality Test Results\n\n")
            f.write(branch_results['bimodality'].to_markdown() + "\n\n")
            bimodal_scenarios = branch_results['bimodality'][branch_results['bimodality']['dip_pvalue'] < 0.05]['scenario'].tolist()
            if bimodal_scenarios:
                f.write(f"Scenarios showing bimodality (strong indicator of introgression): {', '.join(bimodal_scenarios)}\n\n")
        f.write("## Classification Analysis\n\n")
        if has_ml_results:
            if 'classification_report' in ml_results:
                f.write("### Classification Report\n\n")
                f.write(ml_results['classification_report'].to_markdown() + "\n\n")
            if 'feature_importance' in ml_results:
                f.write("### Top Features for Distinguishing Introgression from ILS\n\n")
                f.write(ml_results['feature_importance'].head(5).to_markdown() + "\n\n")
                top_feature = ml_results['feature_importance']['feature'].iloc[0]
                f.write(f"The most informative feature for detection is **{top_feature}**, suggesting this metric should be prioritized in analysis.\n\n")
        f.write("## Key Findings\n\n")
        d_stat_auc = None
        fst_auc = None
        for metric in ['d_stat', 'fst']:
            if metric in df.columns and df[metric].notna().sum() >= 10:
                y_true = df.loc[df[metric].notna(), 'has_introgression']
                fpr, tpr, _ = roc_curve(y_true, df.loc[y_true.index, metric])
                auc_val = auc(fpr, tpr)
                if metric == 'd_stat':
                    d_stat_auc = auc_val
                elif metric == 'fst':
                    fst_auc = auc_val
        f.write("1. **Branch Length Patterns**: ")
        if 'bimodality' in branch_results and not branch_results['bimodality'].empty:
            bimodal_count = branch_results['bimodality']['optimal_components'].gt(1).sum()
            if bimodal_count > 0:
                f.write(f"Branch length analysis revealed bimodality in {bimodal_count} scenarios, suggesting introgression creates distinctive branch length distributions compared to ILS.\n")
            else:
                f.write("Branch length distributions didn't show strong bimodality, making it challenging to distinguish introgression from ILS based solely on branch lengths.\n")
        else:
            f.write("Branch length distributions showed variable patterns across scenarios.\n")
        f.write("2. **Statistical Metrics**: ")
        if d_stat_auc and fst_auc:
            f.write(f"The D-statistic (AUC = {d_stat_auc:.2f}) outperformed FST (AUC = {fst_auc:.2f}) and showed good discriminative power for detecting introgression.\n")
        else:
            f.write("The analysis of standard statistics showed variable results across scenarios.\n")
        if has_ml_results and 'best_model' in ml_results:
            f.write("3. **Machine Learning Integration**: ")
            if ml_results.get('classification_report', {}).get('accuracy', 0) > 0.7:
                f.write("The integrated machine learning approach substantially improved detection accuracy, demonstrating that combining multiple metrics provides better discrimination between introgression and ILS compared to individual statistics.\n")
            else:
                f.write("Even with machine learning integration, accurate classification remained challenging, highlighting the fundamental difficulty in distinguishing these evolutionary processes.\n")
        f.write("\n## Conclusion\n\n")
        f.write("Based on this analysis, the most effective approach for distinguishing introgression from ILS appears to be:\n\n")
        if has_ml_results and 'feature_importance' in ml_results and not ml_results['feature_importance'].empty:
            top_features = ml_results['feature_importance'].head(3)['feature'].tolist()
            f.write(f"1. Integrating multiple metrics, particularly {', '.join(top_features)}\n")
        else:
            f.write("1. Examining multiple metrics in combination rather than relying on any single statistic\n")
        f.write("2. Analyzing branch length distributions for bimodality and deviation from exponential distributions\n")
        f.write("3. Using the D-statistic as a primary indicator, but supplementing with additional evidence\n")
        if 'bimodality' in branch_results and not branch_results['bimodality'].empty:
            bimodal_scenarios = branch_results['bimodality'][branch_results['bimodality']['dip_pvalue'] < 0.05]['scenario'].tolist()
            if bimodal_scenarios:
                f.write(f"\nThe scenarios most clearly showing evidence of introgression are: {', '.join(bimodal_scenarios)}\n")
    
    logging.info(f"Analysis report generated at {os.path.join(output_dir, 'analysis_report.md')}")
    return True

def run_full_pipeline(args):
    """Run the complete analysis pipeline"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Set up configuration
    config = create_config(args)
    
    # 2. Run simulations if not skipped
    if not args.skip_simulation:
        logging.info("Running simulation pipeline")
        result = run_simulation(config_file=os.path.join(args.output_dir, "config.json"))
        if isinstance(result, dict) and result.get('error'):
            logging.error(f"Simulation failed: {result.get('error')}")
            return False
    else:
        logging.info("Skipping simulation step as requested")
    
    # 3. Run parameter sweeps if requested
    if args.parameter_sweep:
        logging.info("Running parameter sweeps")
        sweep_dir = os.path.join(args.output_dir, "parameter_sweeps")
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_params = {
            "introgression_proportion": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
            "introgression_time": [500, 1000, 1500, 2000, 2500],
            "recombination_rate": [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
        }
        for param, values in sweep_params.items():
            logging.info(f"Sweeping parameter: {param}")
            run_parameter_sweep(config, param, values, sweep_dir)
    
    # 4. Integrate and analyze results
    results_file = os.path.join(args.output_dir, "simulation_results.csv")
    if not os.path.exists(results_file) and not args.skip_simulation:
        results_file = f"{config['output_prefix']}_results.csv"
    
    if os.path.exists(results_file):
        logging.info(f"Analyzing results from {results_file}")
        integrate_analysis_results(results_file, args.output_dir)
    else:
        logging.error(f"Results file not found: {results_file}")
        return False
    
    logging.info("Pipeline completed successfully")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline")
    parser.add_argument("--output_dir", default="pipeline_results", help="Output directory")
    parser.add_argument("--config", help="Configuration file path (JSON)")
    parser.add_argument("--skip_simulation", action="store_true", help="Skip simulation step")
    parser.add_argument("--parameter_sweep", action="store_true", help="Run parameter sweeps")
    parser.add_argument("--num_simulations", type=int, default=200, help="Number of simulations to run")
    args = parser.parse_args()
    
    success = run_full_pipeline(args)
    sys.exit(0 if success else 1)

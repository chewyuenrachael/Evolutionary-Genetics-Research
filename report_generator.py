#!/usr/bin/env python3
"""
Comprehensive Report Generator
This script generates a consolidated report summarizing the findings from
all analyses related to distinguishing introgression from ILS.
"""

import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import re
import json
import markdown
import numpy as np
from sklearn.metrics import roc_curve, auc

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("report_generator.log"),
                        logging.StreamHandler()
                    ])

def generate_consolidated_report(results_dir, output_dir, report_title="Introgression vs ILS Analysis Report"):
    """
    Generate a comprehensive report synthesizing all analysis results
    
    Parameters:
    -----------
    results_dir : str
        Directory containing analysis results
    output_dir : str
        Directory to save the report
    report_title : str
        Title for the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report content
    report_content = f"# {report_title}\n\n"
    report_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Find results files
    results_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv') and 'result' in file.lower():
                results_files.append(os.path.join(root, file))
    
    if not results_files:
        logging.warning(f"No results files found in {results_dir}")
        report_content += "## No Results Found\n\n"
        report_content += f"No analysis results were found in the specified directory: {results_dir}\n"
    else:
        # Load primary results file (assume the first one is the main one)
        main_results_file = results_files[0]
        try:
            df = pd.read_csv(main_results_file)
            logging.info(f"Loaded main results file: {main_results_file}")
            
            # 1. Executive Summary
            report_content += "## Executive Summary\n\n"
            
            # Calculate success rates for different metrics
            metrics = ['d_stat', 'fst', 'topology_concordance']
            metric_aucs = {}
            
            for metric in metrics:
                if metric in df.columns:
                    # Create 'has_introgression' column if needed
                    if 'has_introgression' not in df.columns:
                        df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
                    
                    # Calculate ROC curve and AUC
                    valid_idx = df[[metric, 'has_introgression']].dropna().index
                    if len(valid_idx) > 10:
                        metric_values = df.loc[valid_idx, metric]
                        # Invert values for metrics where lower values indicate introgression
                        if metric == 'topology_concordance':
                            metric_values = -metric_values
                            
                        fpr, tpr, _ = roc_curve(df.loc[valid_idx, 'has_introgression'], metric_values)
                        metric_auc = auc(fpr, tpr)
                        metric_aucs[metric] = metric_auc
            
            # Generate summary based on AUC values
            if metric_aucs:
                best_metric = max(metric_aucs.items(), key=lambda x: x[1])
                worst_metric = min(metric_aucs.items(), key=lambda x: x[1])
                
                report_content += "### Key Findings\n\n"
                
                report_content += "**Discriminatory Power of Different Metrics:**\n\n"
                for metric, auc_value in metric_aucs.items():
                    effectiveness = "excellent" if auc_value > 0.9 else "good" if auc_value > 0.7 else "moderate" if auc_value > 0.6 else "poor"
                    report_content += f"- **{metric}**: {auc_value:.2f} AUC ({effectiveness} discriminatory power)\n"
                
                report_content += "\n**Best Discriminator:** "
                if best_metric[1] > 0.7:
                    report_content += f"The {best_metric[0]} metric showed the strongest ability to distinguish between introgression and ILS with an AUC of {best_metric[1]:.2f}.\n\n"
                else:
                    report_content += f"Even the best metric ({best_metric[0]}, AUC={best_metric[1]:.2f}) showed limited discriminatory power, suggesting that these processes are inherently difficult to distinguish based on single metrics alone.\n\n"
                
                report_content += "**Recommendations:**\n\n"
                if best_metric[1] > 0.7:
                    report_content += f"1. The {best_metric[0]} metric should be prioritized when analyzing genetic data for evidence of introgression vs ILS.\n"
                    report_content += "2. Combining multiple metrics through machine learning approaches further improves discrimination.\n"
                    report_content += "3. Branch length distribution analysis provides additional supporting evidence, particularly looking for bimodality.\n"
                else:
                    report_content += "1. No single metric provides reliable discrimination between introgression and ILS in the tested parameter space.\n"
                    report_content += "2. An integrated approach combining multiple lines of evidence is essential.\n"
                    report_content += "3. Additional contextual information (e.g., geographical distribution, phenotypic evidence) should be considered alongside genetic evidence.\n"
            else:
                report_content += "Could not calculate discriminatory power of metrics due to insufficient data.\n\n"
            
            # 2. Detailed Analysis
            report_content += "\n## Detailed Analysis\n\n"
            
            # 2.1. Sample Information
            report_content += "### Dataset Overview\n\n"
            report_content += f"- **Total Simulations:** {len(df)}\n"
            if 'scenario' in df.columns:
                scenarios = df['scenario'].value_counts().to_dict()
                report_content += "- **Scenarios:**\n"
                for scenario, count in scenarios.items():
                    report_content += f"  - {scenario}: {count} simulations\n"
            
            # 2.2. Branch Length Analysis
            report_content += "\n### Branch Length Analysis\n\n"
            
            branch_cols = [col for col in df.columns if 'branch' in col.lower()]
            if branch_cols:
                report_content += "Branch length analysis provides critical insights into distinguishing introgression from ILS, as these processes leave different signatures in the distribution of branch lengths.\n\n"
                
                # Check for existing branch length analysis files
                branch_analysis_files = glob.glob(os.path.join(results_dir, "**/branch_*.png"), recursive=True)
                if branch_analysis_files:
                    report_content += "#### Branch Length Distribution Patterns\n\n"
                    
                    # Include a sample visualization
                    vis_file = next((f for f in branch_analysis_files if "distribution" in f.lower() or "comparison" in f.lower()), branch_analysis_files[0])
                    vis_rel_path = os.path.relpath(vis_file, output_dir)
                    report_content += f"![Branch Length Analysis]({vis_rel_path})\n\n"
                    
                    # Describe patterns based on branch length statistics
                    if 'has_introgression' in df.columns and 'mean_internal_branch' in df.columns:
                        ils_branches = df[~df['has_introgression']]['mean_internal_branch'].dropna()
                        intro_branches = df[df['has_introgression']]['mean_internal_branch'].dropna()
                        
                        if len(ils_branches) > 5 and len(intro_branches) > 5:
                            mean_diff = intro_branches.mean() - ils_branches.mean()
                            direction = "longer" if mean_diff > 0 else "shorter"
                            
                            report_content += f"On average, internal branches in introgression scenarios are {direction} "
                            report_content += f"than in ILS scenarios ({intro_branches.mean():.2f} vs {ils_branches.mean():.2f}).\n\n"
                            
                            # Check variance difference
                            var_diff = intro_branches.var() - ils_branches.var()
                            var_desc = "more variable" if var_diff > 0 else "less variable"
                            
                            report_content += f"Branch lengths in introgression scenarios also tend to be {var_desc} "
                            report_content += f"({intro_branches.var():.2f} vs {ils_branches.var():.2f}), "
                            
                            if var_diff > 0:
                                report_content += "which is consistent with the theoretically expected bimodal distribution pattern.\n\n"
                            else:
                                report_content += "which is somewhat contrary to theoretical expectations of higher variability under introgression.\n\n"
                
                # Check for branch ratio analysis
                if 'min_internal_branch' in df.columns and 'max_internal_branch' in df.columns:
                    report_content += "#### Branch Length Ratio Analysis\n\n"
                    
                    df['branch_ratio'] = df['max_internal_branch'] / df['min_internal_branch']
                    
                    report_content += "The ratio between maximum and minimum internal branch lengths provides a useful metric for distinguishing introgression from ILS:\n\n"
                    
                    if 'has_introgression' in df.columns:
                        ils_ratio = df[~df['has_introgression']]['branch_ratio'].dropna()
                        intro_ratio = df[df['has_introgression']]['branch_ratio'].dropna()
                        
                        if len(ils_ratio) > 5 and len(intro_ratio) > 5:
                            report_content += f"- Introgression scenarios: mean ratio = {intro_ratio.mean():.2f}\n"
                            report_content += f"- ILS scenarios: mean ratio = {ils_ratio.mean():.2f}\n\n"
                            
                            ratio_diff = intro_ratio.mean() - ils_ratio.mean()
                            direction = "higher" if ratio_diff > 0 else "lower"
                            
                            report_content += f"The {direction} branch length ratio in introgression scenarios reflects "
                            
                            if ratio_diff > 0:
                                report_content += "the presence of distinct coalescent events at different time points, consistent with gene flow between diverged lineages.\n\n"
                            else:
                                report_content += "an unexpected pattern that warrants further investigation.\n\n"
            else:
                report_content += "Branch length data not available in the analysis results.\n\n"
            
            # 2.3. Statistical Metrics Analysis
            report_content += "\n### Statistical Metrics\n\n"
            
            metric_cols = ['fst', 'd_stat', 'topology_concordance']
            available_metrics = [m for m in metric_cols if m in df.columns]
            
            if available_metrics:
                report_content += "Several statistical metrics were evaluated for their ability to distinguish between introgression and ILS:\n\n"
                
                for metric in available_metrics:
                    report_content += f"#### {metric.upper()}\n\n"
                    
                    if 'has_introgression' in df.columns:
                        ils_values = df[~df['has_introgression']][metric].dropna()
                        intro_values = df[df['has_introgression']][metric].dropna()
                        
                        if len(ils_values) > 5 and len(intro_values) > 5:
                            # Calculate stats
                            ils_mean = ils_values.mean()
                            intro_mean = intro_values.mean()
                            mean_diff = intro_mean - ils_mean
                            direction = "higher" if mean_diff > 0 else "lower"
                            
                            report_content += f"- Mean value in introgression scenarios: {intro_mean:.4f}\n"
                            report_content += f"- Mean value in ILS scenarios: {ils_mean:.4f}\n\n"
                            
                            # Describe the pattern
                            report_content += f"The {metric.upper()} values are, on average, {direction} in introgression scenarios. "
                            
                            # Metric-specific interpretations
                            if metric == 'd_stat':
                                if mean_diff > 0:
                                    report_content += "This is expected as the D-statistic is specifically designed to detect gene flow between non-sister taxa.\n\n"
                                else:
                                    report_content += "This is contrary to expectations, as the D-statistic should typically be elevated in the presence of gene flow.\n\n"
                            elif metric == 'fst':
                                if mean_diff < 0:
                                    report_content += "Lower FST in introgression scenarios reflects increased genetic similarity due to gene flow.\n\n"
                                else:
                                    report_content += "Higher FST in introgression scenarios is unexpected and may reflect complex demographic dynamics.\n\n"
                            elif metric == 'topology_concordance':
                                if mean_diff < 0:
                                    report_content += "Lower topology concordance in introgression scenarios reflects the disruption of species tree patterns by gene flow.\n\n"
                                else:
                                    report_content += "Higher topology concordance in introgression scenarios is unexpected and warrants further investigation.\n\n"
                    else:
                        report_content += f"Unable to compare {metric} between introgression and ILS scenarios due to missing classification.\n\n"
            else:
                report_content += "No standard statistical metrics (FST, D-statistic, etc.) were found in the analysis results.\n\n"
            
            # 2.4. Machine Learning Insights
            report_content += "\n### Machine Learning Insights\n\n"
            
            # Check for feature importance files
            feature_files = glob.glob(os.path.join(results_dir, "**/feature_*.csv"), recursive=True) + \
                           glob.glob(os.path.join(results_dir, "**/feature_*.png"), recursive=True)
            
            if feature_files:
                report_content += "Machine learning approaches provide valuable insights by integrating multiple metrics and identifying the most informative features for distinguishing introgression from ILS.\n\n"
                
                # Include feature importance visualization if available
                vis_file = next((f for f in feature_files if f.endswith('.png')), None)
                if vis_file:
                    vis_rel_path = os.path.relpath(vis_file, output_dir)
                    report_content += f"![Feature Importance]({vis_rel_path})\n\n"
                
                # Include feature importance data if available
                data_file = next((f for f in feature_files if f.endswith('.csv')), None)
                if data_file:
                    try:
                        feature_data = pd.read_csv(data_file)
                        if 'feature' in feature_data.columns and 'importance' in feature_data.columns:
                            # Sort by importance
                            feature_data = feature_data.sort_values('importance', ascending=False)
                            
                            report_content += "#### Key Features for Discrimination\n\n"
                            report_content += "The machine learning analysis identified the following features as most informative:\n\n"
                            
                            # List top features
                            top_n = min(5, len(feature_data))
                            for i in range(top_n):
                                feature = feature_data['feature'].iloc[i]
                                importance = feature_data['importance'].iloc[i]
                                report_content += f"{i+1}. **{feature}** (importance: {importance:.4f})\n"
                            
                            report_content += "\n"
                    except Exception as e:
                        logging.error(f"Error processing feature importance data: {str(e)}")
                        report_content += "Error processing feature importance data.\n\n"
            else:
                report_content += "No machine learning analysis results were found.\n\n"
            
            # 2.5. Parameter Sweep Insights
            report_content += "\n### Parameter Sensitivity Analysis\n\n"
            
            # Check for parameter sweep files
            sweep_files = glob.glob(os.path.join(results_dir, "**/sweep_*.csv"), recursive=True) + \
                         glob.glob(os.path.join(results_dir, "**/parameter_*.png"), recursive=True)
            
            if sweep_files:
                report_content += "Parameter sweep analysis reveals how the ability to distinguish introgression from ILS varies across different evolutionary scenarios and parameter combinations.\n\n"
                
                # Include parameter sweep visualization if available
                vis_file = next((f for f in sweep_files if f.endswith('.png')), None)
                if vis_file:
                    vis_rel_path = os.path.relpath(vis_file, output_dir)
                    report_content += f"![Parameter Sweep]({vis_rel_path})\n\n"
                
                # Extract key insights from parameter sweeps
                sweep_data_files = [f for f in sweep_files if f.endswith('.csv')]
                if sweep_data_files:
                    report_content += "#### Key Parameter Insights\n\n"
                    
                    for sweep_file in sweep_data_files[:3]:  # Limit to first 3 files
                        try:
                            sweep_data = pd.read_csv(sweep_file)
                            
                            # Extract parameter names
                            param_cols = [col for col in sweep_data.columns 
                                         if col not in ['avg_d_stat', 'avg_fst', 'd_stat_auc', 'num_samples']]
                            
                            if len(param_cols) > 0 and 'd_stat_auc' in sweep_data.columns:
                                param_name = param_cols[0]
                                
                                # Calculate correlation
                                corr = sweep_data[[param_name, 'd_stat_auc']].corr().iloc[0, 1]
                                
                                # Describe relationship
                                relationship = "positive" if corr > 0.3 else "negative" if corr < -0.3 else "weak"
                                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "slight"
                                
                                report_content += f"- **{param_name}**: There is a {strength} {relationship} relationship (correlation: {corr:.2f}) between this parameter and the ability to distinguish introgression from ILS.\n"
                                
                                if 'introgression_time' in param_name and corr < -0.3:
                                    report_content += "  - More recent introgression events are easier to detect, as expected due to less time for subsequent evolutionary processes to obscure the signal.\n"
                                elif 'introgression_proportion' in param_name and corr > 0.3:
                                    report_content += "  - Higher rates of gene flow produce stronger signals that are easier to distinguish from ILS.\n"
                                elif 'recombination_rate' in param_name:
                                    if corr > 0.3:
                                        report_content += "  - Higher recombination rates appear to enhance the detectability of introgression, possibly by creating more distinct haplotype blocks.\n"
                                    elif corr < -0.3:
                                        report_content += "  - Higher recombination rates seem to obscure introgression signals, possibly by breaking up introgressed segments.\n"
                            
                        except Exception as e:
                            logging.error(f"Error processing sweep file {sweep_file}: {str(e)}")
                    
                    report_content += "\n"
            else:
                report_content += "No parameter sweep analysis results were found.\n\n"
            
            # 3. Recommendations
            report_content += "\n## Recommendations\n\n"
            
            report_content += "Based on the comprehensive analysis of simulated data, we recommend the following approach for distinguishing introgression from incomplete lineage sorting (ILS) in empirical studies:\n\n"
            
            # Generate recommendations based on analysis results
            report_content += "### 1. Analytical Approach\n\n"
            
            if metric_aucs:
                best_metric = max(metric_aucs.items(), key=lambda x: x[1])
                if best_metric[1] > 0.7:
                    report_content += f"- Prioritize the {best_metric[0]} metric as the primary indicator of introgression.\n"
                else:
                    report_content += "- Use multiple complementary metrics rather than relying on any single statistic.\n"
            else:
                report_content += "- Use multiple complementary metrics rather than relying on any single statistic.\n"
            
            report_content += "- Examine branch length distributions for bimodality, which can be indicative of introgression.\n"
            report_content += "- Calculate branch length ratios (max/min) to help distinguish between the processes.\n"
            report_content += "- When possible, apply machine learning approaches that integrate multiple metrics.\n\n"
            
            report_content += "### 2. Sampling and Data Collection\n\n"
            report_content += "- Maximize sequence length and genomic coverage to improve statistical power.\n"
            report_content += "- Include appropriate outgroups to polarize derived alleles for D-statistic calculation.\n"
            report_content += "- Sample multiple individuals per population to improve allele frequency estimates.\n\n"
            
            report_content += "### 3. Interpretation Guidelines\n\n"
            report_content += "- Consider that detection power varies with:\n"
            report_content += "  - Time since introgression (more recent events are generally easier to detect)\n"
            report_content += "  - Proportion of introgression (higher gene flow produces stronger signals)\n"
            report_content += "  - Background recombination rate (affects the preservation of introgression signals)\n"
            report_content += "- Integrate additional contextual information, such as geographical distribution and phenotypic evidence.\n"
            report_content += "- Acknowledge uncertainty in cases where signals are ambiguous.\n\n"
            
            # 4. Limitations and Future Directions
            report_content += "\n## Limitations and Future Directions\n\n"
            
            report_content += "### Limitations of Current Approach\n\n"
            report_content += "- The simulations assume simplified demographic models that may not capture all complexities of real evolutionary histories.\n"
            report_content += "- The analysis focuses primarily on neutral evolution and does not fully account for the effects of selection.\n"
            report_content += "- The approach assumes discrete introgression events rather than continuous gene flow or multiple introgression episodes.\n\n"
            
            report_content += "### Future Directions\n\n"
            report_content += "- Expand simulations to include more complex demographic scenarios, including multiple introgression events.\n"
            report_content += "- Incorporate the effects of selection, particularly for adaptive introgression.\n"
            report_content += "- Develop methods that can estimate the timing and magnitude of introgression more precisely.\n"
            report_content += "- Integrate with other sources of evidence, such as morphological data and biogeography.\n"
            
        except Exception as e:
            logging.error(f"Error analyzing results file: {str(e)}")
            report_content += f"## Error Processing Results\n\n"
            report_content += f"An error occurred while processing the results file: {str(e)}\n"
    
    # Write report to file
    report_path = os.path.join(output_dir, "consolidated_report.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Convert to HTML if possible
    try:
        html_content = markdown.markdown(report_content)
        html_path = os.path.join(output_dir, "consolidated_report.html")
        
        # Add simple HTML wrapper with some basic styling
        html_full = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                    border: 1px solid #ddd;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_full)
        
        logging.info(f"HTML report generated at {html_path}")
    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")
    
    logging.info(f"Consolidated report generated at {report_path}")
    
    return report_path

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate consolidated report")
    parser.add_argument("--results_dir", required=True, help="Directory containing analysis results")
    parser.add_argument("--output_dir", default="consolidated_report", help="Directory to save the report")
    parser.add_argument("--title", default="Introgression vs ILS Analysis Report", help="Report title")
    args = parser.parse_args()
    
    # Generate report
    report_path = generate_consolidated_report(args.results_dir, args.output_dir, args.title)
    
    print(f"\nReport generated successfully at: {report_path}")
    print(f"HTML version (if available): {os.path.join(args.output_dir, 'consolidated_report.html')}")

if __name__ == "__main__":
    main()
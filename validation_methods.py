#!/usr/bin/env python3
"""
Validation Methods Module for Evolutionary Genetics
Testing the robustness of introgression vs ILS differentiation
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Tuple, Optional, Union
import logging
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ParameterRecoveryTest:
    """
    Test ability to recover parameters from simulated data
    """
    
    def __init__(self, pipeline_script: str, 
                output_dir: str = "parameter_recovery",
                n_simulations: int = 20):
        """
        Initialize parameter recovery test
        
        Parameters:
        -----------
        pipeline_script : str
            Path to pipeline script
        output_dir : str
            Directory to save results
        n_simulations : int
            Number of simulations to run for each test
        """
        self.pipeline_script = pipeline_script
        self.output_dir = output_dir
        self.n_simulations = n_simulations
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_pipeline_with_params(self, params: Dict) -> str:
        """
        Run pipeline with specific parameters
        
        Parameters:
        -----------
        params : Dict
            Pipeline parameters
            
        Returns:
        --------
        str
            Path to results file
        """
        # Create config file
        config_file = os.path.join(self.output_dir, f"config_{random.randint(1000, 9999)}.json")
        with open(config_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Run pipeline script
        logging.info(f"Running pipeline with parameters: {params}")
        output_prefix = params.get('output_prefix', 'recovery_test')
        
        try:
            cmd = ["python", self.pipeline_script, "--config", config_file]
            subprocess.run(cmd, check=True)
            
            # Return path to expected results file
            return f"{output_prefix}_results.csv"
        except subprocess.CalledProcessError as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            return None
    
    def test_introgression_parameter_recovery(self, 
                                            param_ranges: Dict = None) -> pd.DataFrame:
        """
        Test recovery of introgression parameters
        
        Parameters:
        -----------
        param_ranges : Dict
            Ranges for parameters to test
            
        Returns:
        --------
        pd.DataFrame
            Test results
        """
        if param_ranges is None:
            # Default parameter ranges
            param_ranges = {
                'introgression_time': [500, 1000, 1500, 2000],
                'introgression_proportion': [0.05, 0.1, 0.15, 0.2],
                'recombination_rate': [1e-9, 1e-8, 1e-7]
            }
        
        # Create grid of parameter combinations
        param_grid = list(ParameterGrid(param_ranges))
        
        results = []
        
        for i, params in enumerate(param_grid):
            logging.info(f"Running test {i+1}/{len(param_grid)}")
            
            # Create pipeline config
            pipeline_config = {
                'num_simulations': self.n_simulations,
                'genome_length': 1e5,
                'max_workers': min(os.cpu_count(), 4),
                'output_prefix': f"recovery_test_{i}",
                'save_trees': True,
                'scenario_weights': {
                    'base': 1.0,  # Only use base scenario for recovery tests
                    'rapid_radiation': 0.0,
                    'bottleneck': 0.0,
                    'continuous_migration': 0.0
                }
            }
            
            # Set fixed parameters
            for param, value in params.items():
                if param == 'introgression_time':
                    pipeline_config['introgression_time_range'] = [value, value]
                elif param == 'introgression_proportion':
                    pipeline_config['introgression_proportion_range'] = [value, value]
                elif param == 'recombination_rate':
                    pipeline_config['recombination_rate_range'] = [value, value]
            
            # Run pipeline
            results_file = self.run_pipeline_with_params(pipeline_config)
            
            if results_file and os.path.exists(results_file):
                # Load results
                df = pd.read_csv(results_file)
                
                # Calculate summary statistics
                d_stat_mean = df['d_stat'].mean()
                d_stat_std = df['d_stat'].std()
                fst_mean = df['fst'].mean()
                fst_std = df['fst'].std()
                
                # Store results
                results.append({
                    'test_id': i,
                    **params,
                    'd_stat_mean': d_stat_mean,
                    'd_stat_std': d_stat_std,
                    'fst_mean': fst_mean,
                    'fst_std': fst_std,
                    'n_simulations': len(df)
                })
        
        # Combine results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = os.path.join(self.output_dir, "parameter_recovery_results.csv")
        results_df.to_csv(results_file, index=False)
        
        logging.info(f"Parameter recovery test completed, results saved to {results_file}")
        
        return results_df
    
    def analyze_recovery_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze parameter recovery test results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Test results
            
        Returns:
        --------
        Dict
            Analysis results
        """
        # Check if results exist
        if len(results_df) == 0:
            return {'error': 'No results to analyze'}
        
        analysis = {}
        
        # Analyze D-statistic correlation with introgression parameters
        for param in ['introgression_time', 'introgression_proportion', 'recombination_rate']:
            if param in results_df.columns:
                # Calculate correlation
                corr, p_value = stats.pearsonr(results_df[param], results_df['d_stat_mean'])
                
                # Store results
                analysis[f'{param}_correlation'] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot D-statistic vs introgression parameters
        for i, param in enumerate(['introgression_time', 'introgression_proportion', 'recombination_rate']):
            if param in results_df.columns:
                plt.subplot(1, 3, i+1)
                
                # Create scatter plot
                sns.scatterplot(
                    x=param, 
                    y='d_stat_mean', 
                    data=results_df,
                    s=50
                )
                
                # Add error bars
                plt.errorbar(
                    results_df[param], 
                    results_df['d_stat_mean'], 
                    yerr=results_df['d_stat_std'],
                    fmt='none', 
                    ecolor='gray', 
                    alpha=0.5
                )
                
                # Add regression line
                sns.regplot(
                    x=param, 
                    y='d_stat_mean', 
                    data=results_df,
                    scatter=False,
                    line_kws={'color': 'red'}
                )
                
                # Add labels
                plt.xlabel(param.replace('_', ' ').title())
                plt.ylabel('D-statistic (mean)')
                plt.title(f'D-statistic vs {param}')
                
                # Use log scale for rates
                if 'rate' in param:
                    plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recovery_analysis.png"), dpi=300)
        plt.close()
        
        # Save analysis
        with open(os.path.join(self.output_dir, "recovery_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logging.info(f"Recovery analysis completed")
        
        return analysis

class StatisticalPowerTest:
    """
    Test statistical power for differentiating introgression from ILS
    """
    
    def __init__(self, pipeline_script: str, 
                output_dir: str = "power_test",
                n_repeats: int = 10):
        """
        Initialize statistical power test
        
        Parameters:
        -----------
        pipeline_script : str
            Path to pipeline script
        output_dir : str
            Directory to save results
        n_repeats : int
            Number of times to repeat each test condition
        """
        self.pipeline_script = pipeline_script
        self.output_dir = output_dir
        self.n_repeats = n_repeats
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_power_test(self, 
                      num_simulations: int = 50,
                      param_variations: Dict = None) -> pd.DataFrame:
        """
        Run statistical power test with varying parameters
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations per run
        param_variations : Dict
            Parameter variations to test
            
        Returns:
        --------
        pd.DataFrame
            Test results
        """
        if param_variations is None:
            # Default parameter variations
            param_variations = {
                'introgression_proportion': [0.01, 0.05, 0.1, 0.15, 0.2],
                'num_simulations': [num_simulations]
            }
        
        # Create parameter grid
        param_grid = list(ParameterGrid(param_variations))
        
        # Initialize results
        results = []
        
        # Run tests in parallel
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = []
            
            for i, params in enumerate(param_grid):
                for rep in range(self.n_repeats):
                    # Create test configuration
                    test_config = self._create_test_config(
                        params, rep, num_simulations=params.get('num_simulations', num_simulations)
                    )
                    
                    # Submit task
                    futures.append(executor.submit(
                        self._run_single_test, i, rep, test_config, params
                    ))
            
            # Collect results
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        
        # Combine results
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = os.path.join(self.output_dir, "power_test_results.csv")
        results_df.to_csv(results_file, index=False)
        
        logging.info(f"Power test completed with {len(results_df)} tests")
        
        return results_df
    
    def _create_test_config(self, params: Dict, rep: int, num_simulations: int) -> Dict:
        """Create configuration for a single test"""
        # Base configuration
        config = {
            'num_simulations': num_simulations,
            'genome_length': 1e5,
            'max_workers': min(os.cpu_count(), 4),
            'output_prefix': f"power_test_{rep}",
            'save_trees': False
        }
        
        # Set introgression proportion if specified
        if 'introgression_proportion' in params:
            prop = params['introgression_proportion']
            config['scenario_weights'] = {
                'base': 0.5,
                'rapid_radiation': 0.0,
                'bottleneck': 0.0,
                'continuous_migration': 0.5
            }
            config['introgression_proportion_range'] = [prop, prop]
        
        return config
    
    def _run_single_test(self, test_id: int, rep: int, config: Dict, params: Dict) -> Dict:
        """Run a single power test"""
        logging.info(f"Running power test {test_id}, repeat {rep}")
        
        try:
            # Create config file
            config_file = os.path.join(self.output_dir, f"config_{test_id}_{rep}.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run pipeline
            cmd = ["python", self.pipeline_script, "--config", config_file]
            subprocess.run(cmd, check=True)
            
            # Load results
            results_file = f"{config['output_prefix']}_results.csv"
            if not os.path.exists(results_file):
                logging.error(f"Results file not found: {results_file}")
                return None
                
            df = pd.read_csv(results_file)
            
            # Calculate statistical power
            # Define true labels (0 = ILS/other, 1 = introgression)
            has_introgression = ~df['scenario'].isin(['continuous_migration'])
            
            # Calculate AUC for D-statistic
            if 'd_stat' in df.columns:
                fpr, tpr, _ = roc_curve(has_introgression, df['d_stat'])
                d_stat_auc = auc(fpr, tpr)
            else:
                d_stat_auc = None
            
            # Calculate AUC for FST
            if 'fst' in df.columns:
                fpr, tpr, _ = roc_curve(has_introgression, df['fst'])
                fst_auc = auc(fpr, tpr)
            else:
                fst_auc = None
            
            # Calculate AUC for branch length metrics
            if 'mean_internal_branch' in df.columns:
                fpr, tpr, _ = roc_curve(has_introgression, -df['mean_internal_branch'])
                branch_auc = auc(fpr, tpr)
            else:
                branch_auc = None
            
            # Return results
            result = {
                'test_id': test_id,
                'repeat': rep,
                **params,
                'n_simulations': len(df),
                'd_stat_auc': d_stat_auc,
                'fst_auc': fst_auc,
                'branch_auc': branch_auc
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Power test failed: {str(e)}")
            return None
    
    def analyze_power_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze power test results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Power test results
            
        Returns:
        --------
        Dict
            Analysis results
        """
        # Check if results exist
        if len(results_df) == 0:
            return {'error': 'No results to analyze'}
        
        analysis = {}
        
        # Calculate average AUC by parameter
        grouped_results = results_df.groupby('introgression_proportion').agg({
            'd_stat_auc': ['mean', 'std'],
            'fst_auc': ['mean', 'std'],
            'branch_auc': ['mean', 'std'],
            'n_simulations': 'mean'
        }).reset_index()
        
        # Save grouped results
        grouped_file = os.path.join(self.output_dir, "power_test_summary.csv")
        grouped_results.to_csv(grouped_file)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot AUC vs introgression proportion
        metrics = ['d_stat_auc', 'fst_auc', 'branch_auc']
        metric_labels = ['D-statistic', 'FST', 'Branch Length']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            if f"{metric}" in grouped_results.columns:
                # Extract mean and std
                mean_col = (metric, 'mean')
                std_col = (metric, 'std')
                
                # Plot with error bars
                plt.errorbar(
                    grouped_results['introgression_proportion'], 
                    grouped_results[mean_col],
                    yerr=grouped_results[std_col],
                    fmt='-o',
                    color=color,
                    label=label,
                    linewidth=2,
                    markersize=8
                )
        
        # Add reference line for random classifier (AUC = 0.5)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Introgression Proportion')
        plt.ylabel('Area Under ROC Curve (AUC)')
        plt.title('Statistical Power for Detecting Introgression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "power_analysis.png"), dpi=300)
        plt.close()
        
        # Determine optimal thresholds for each metric
        optimal_thresholds = {}
        
        for metric in metrics:
            optimal_thresholds[metric] = {}
            
            for introgression_prop in results_df['introgression_proportion'].unique():
                subset = results_df[results_df['introgression_proportion'] == introgression_prop]
                
                if metric in subset.columns:
                    mean_auc = subset[metric].mean()
                    optimal_thresholds[metric][float(introgression_prop)] = {
                        'auc': float(mean_auc),
                        'power': float(mean_auc > 0.7)  # AUC > 0.7 indicates good discrimination
                    }
        
        # Store in analysis
        analysis['grouped_results'] = grouped_results.to_dict()
        analysis['optimal_thresholds'] = optimal_thresholds
        
        # Save analysis
        with open(os.path.join(self.output_dir, "power_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logging.info(f"Power analysis completed")
        
        return analysis

class RobustnessTest:
    """
    Test robustness of methods to various perturbations
    """
    
    def __init__(self, pipeline_script: str, 
                output_dir: str = "robustness_test"):
        """
        Initialize robustness test
        
        Parameters:
        -----------
        pipeline_script : str
            Path to pipeline script
        output_dir : str
            Directory to save results
        """
        self.pipeline_script = pipeline_script
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def test_sample_size_robustness(self, 
                                  sample_sizes: List[int] = None) -> pd.DataFrame:
        """
        Test robustness to sample size variation
        
        Parameters:
        -----------
        sample_sizes : List[int]
            List of sample sizes to test
            
        Returns:
        --------
        pd.DataFrame
            Test results
        """
        if sample_sizes is None:
            sample_sizes = [10, 20, 50, 100, 200]
        
        results = []
        
        for size in sample_sizes:
            logging.info(f"Testing sample size: {size}")
            
            # Create configuration
            config = {
                'num_simulations': size,
                'genome_length': 1e5,
                'max_workers': min(os.cpu_count(), 4),
                'output_prefix': f"sample_size_{size}",
                'save_trees': False,
                'scenario_weights': {
                    'base': 0.4,
                    'rapid_radiation': 0.3,
                    'bottleneck': 0.2,
                    'continuous_migration': 0.1
                }
            }
            
            # Create config file
            config_file = os.path.join(self.output_dir, f"config_size_{size}.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            try:
                # Run pipeline
                cmd = ["python", self.pipeline_script, "--config", config_file]
                subprocess.run(cmd, check=True)
                
                # Load results
                results_file = f"{config['output_prefix']}_results.csv"
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    
                    # Calculate metrics
                    d_stat_std = df['d_stat'].std()
                    fst_std = df['fst'].std()
                    
                    # Check if we have branch length metrics
                    if 'mean_internal_branch' in df.columns:
                        branch_std = df['mean_internal_branch'].std()
                    else:
                        branch_std = None
                    
                    # Add to results
                    results.append({
                        'sample_size': size,
                        'n_samples': len(df),
                        'd_stat_std': d_stat_std,
                        'fst_std': fst_std,
                        'branch_std': branch_std
                    })
            
            except Exception as e:
                logging.error(f"Robustness test failed for sample size {size}: {str(e)}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = os.path.join(self.output_dir, "sample_size_robustness.csv")
        results_df.to_csv(results_file, index=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        metrics = ['d_stat_std', 'fst_std', 'branch_std']
        labels = ['D-statistic', 'FST', 'Branch Length']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for metric, label, color in zip(metrics, labels, colors):
            if metric in results_df.columns:
                plt.plot(
                    results_df['sample_size'],
                    results_df[metric],
                    'o-',
                    label=label,
                    color=color,
                    linewidth=2,
                    markersize=8
                )
        
        plt.xlabel('Sample Size')
        plt.ylabel('Standard Deviation')
        plt.title('Method Robustness to Sample Size Variation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sample_size_robustness.png"), dpi=300)
        plt.close()
        
        logging.info(f"Sample size robustness test completed")
        
        return results_df
    
    def test_parameter_robustness(self, parameter_variations: Dict = None) -> pd.DataFrame:
        """
        Test robustness to parameter variations
        
        Parameters:
        -----------
        parameter_variations : Dict
            Dictionary of parameter ranges to test
            
        Returns:
        --------
        pd.DataFrame
            Test results
        """
        if parameter_variations is None:
            parameter_variations = {
                'mutation_rate_range': [(1e-9, 1e-8), (1e-8, 1e-7), (1e-7, 1e-6)],
                'recombination_rate_range': [(1e-10, 1e-9), (1e-9, 1e-8), (1e-8, 1e-7)]
            }
        
        results = []
        
        # Get all combinations of parameter variations
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())
        
        import itertools
        for values in itertools.product(*param_values):
            param_dict = {name: value for name, value in zip(param_names, values)}
            
            param_str = "_".join([f"{k.split('_')[0]}_{v[0]:.0e}" for k, v in param_dict.items()])
            logging.info(f"Testing parameter variation: {param_str}")
            
            # Create configuration
            config = {
                'num_simulations': 50,
                'genome_length': 1e5,
                'max_workers': min(os.cpu_count(), 4),
                'output_prefix': f"param_var_{param_str}",
                'save_trees': False,
                'scenario_weights': {
                    'base': 0.5,
                    'rapid_radiation': 0.0,
                    'bottleneck': 0.0,
                    'continuous_migration': 0.5
                }
            }
            
            # Add parameter variations
            for key, value in param_dict.items():
                config[key] = value
            
            # Create config file
            config_file = os.path.join(self.output_dir, f"config_{param_str}.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            try:
                # Run pipeline
                cmd = ["python", self.pipeline_script, "--config", config_file]
                subprocess.run(cmd, check=True)
                
                # Load results
                results_file = f"{config['output_prefix']}_results.csv"
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    
                    # Define true labels (0 = ILS/other, 1 = introgression)
                    has_introgression = ~df['scenario'].isin(['continuous_migration'])
                    
                    # Calculate AUC for D-statistic
                    if 'd_stat' in df.columns:
                        fpr, tpr, _ = roc_curve(has_introgression, df['d_stat'])
                        d_stat_auc = auc(fpr, tpr)
                    else:
                        d_stat_auc = None
                    
                    # Calculate AUC for FST
                    if 'fst' in df.columns:
                        fpr, tpr, _ = roc_curve(has_introgression, df['fst'])
                        fst_auc = auc(fpr, tpr)
                    else:
                        fst_auc = None
                    
                    # Add to results
                    results.append({
                        **{k: str(v) for k, v in param_dict.items()},
                        'n_samples': len(df),
                        'd_stat_auc': d_stat_auc,
                        'fst_auc': fst_auc
                    })
            
            except Exception as e:
                logging.error(f"Robustness test failed for {param_str}: {str(e)}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = os.path.join(self.output_dir, "parameter_robustness.csv")
        results_df.to_csv(results_file, index=False)
        
        logging.info(f"Parameter robustness test completed")
        
        return results_df
    
    def analyze_robustness_results(self, sample_size_df: pd.DataFrame = None,
                                parameter_df: pd.DataFrame = None) -> Dict:
        """
        Analyze robustness test results
        
        Parameters:
        -----------
        sample_size_df : pd.DataFrame
            Sample size robustness results
        parameter_df : pd.DataFrame
            Parameter robustness results
            
        Returns:
        --------
        Dict
            Analysis results
        """
        analysis = {}
        
        # Analyze sample size robustness
        if sample_size_df is not None and not sample_size_df.empty:
            # Calculate coefficient of variation (CV) for each metric
            for metric in ['d_stat_std', 'fst_std', 'branch_std']:
                if metric in sample_size_df.columns:
                    cv = sample_size_df[metric].std() / sample_size_df[metric].mean()
                    analysis[f"{metric}_cv"] = float(cv)
            
            # Determine optimal sample size (where standard deviation stabilizes)
            optimal_size = None
            for i in range(1, len(sample_size_df)):
                pct_change = abs(sample_size_df['d_stat_std'].iloc[i] - 
                              sample_size_df['d_stat_std'].iloc[i-1]) / sample_size_df['d_stat_std'].iloc[i-1]
                if pct_change < 0.1:  # Less than 10% change
                    optimal_size = sample_size_df['sample_size'].iloc[i]
                    break
            
            analysis['optimal_sample_size'] = optimal_size
        
        # Analyze parameter robustness
        if parameter_df is not None and not parameter_df.empty:
            # Calculate mean AUC across parameter variations
            mean_d_stat_auc = parameter_df['d_stat_auc'].mean()
            mean_fst_auc = parameter_df['fst_auc'].mean()
            
            # Calculate standard deviation of AUC
            std_d_stat_auc = parameter_df['d_stat_auc'].std()
            std_fst_auc = parameter_df['fst_auc'].std()
            
            analysis['d_stat_auc_mean'] = float(mean_d_stat_auc)
            analysis['d_stat_auc_std'] = float(std_d_stat_auc)
            analysis['fst_auc_mean'] = float(mean_fst_auc)
            analysis['fst_auc_std'] = float(std_fst_auc)
            
            # Find parameters with highest AUC
            best_d_stat_idx = parameter_df['d_stat_auc'].idxmax()
            best_fst_idx = parameter_df['fst_auc'].idxmax()
            
            analysis['best_d_stat_params'] = parameter_df.iloc[best_d_stat_idx].to_dict()
            analysis['best_fst_params'] = parameter_df.iloc[best_fst_idx].to_dict()
        
        # Save analysis
        with open(os.path.join(self.output_dir, "robustness_analysis.json"), 'w') as f:
            # Filter out non-serializable objects
            serializable_analysis = {}
            for k, v in analysis.items():
                try:
                    json.dumps({k: v})
                    serializable_analysis[k] = v
                except:
                    serializable_analysis[k] = str(v)
            
            json.dump(serializable_analysis, f, indent=2)
        
        logging.info(f"Robustness analysis completed")
        
        return analysis

def run_all_validation_tests(pipeline_script: str, output_dir: str = "validation_results"):
    """
    Run all validation tests
    
    Parameters:
    -----------
    pipeline_script : str
        Path to pipeline script
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run parameter recovery test
    logging.info("Starting parameter recovery test")
    recovery_dir = os.path.join(output_dir, "parameter_recovery")
    recovery_test = ParameterRecoveryTest(pipeline_script, recovery_dir)
    recovery_results = recovery_test.test_introgression_parameter_recovery()
    recovery_analysis = recovery_test.analyze_recovery_results(recovery_results)
    
    # Run statistical power test
    logging.info("Starting statistical power test")
    power_dir = os.path.join(output_dir, "power_test")
    power_test = StatisticalPowerTest(pipeline_script, power_dir)
    power_results = power_test.run_power_test()
    power_analysis = power_test.analyze_power_results(power_results)
    
    # Run robustness tests
    logging.info("Starting robustness tests")
    robustness_dir = os.path.join(output_dir, "robustness_test")
    robustness_test = RobustnessTest(pipeline_script, robustness_dir)
    
    # Sample size robustness
    sample_size_results = robustness_test.test_sample_size_robustness()
    
    # Parameter robustness
    parameter_results = robustness_test.test_parameter_robustness()
    
    # Analyze robustness
    robustness_analysis = robustness_test.analyze_robustness_results(
        sample_size_results, parameter_results
    )
    
    # Combine all results
    summary = {
        'recovery_test': recovery_analysis,
        'power_test': power_analysis,
        'robustness_test': robustness_analysis
    }
    
    # Save summary
    with open(os.path.join(output_dir, "validation_summary.json"), 'w') as f:
        # Filter out non-serializable objects
        serializable_summary = {}
        for k1, v1 in summary.items():
            serializable_summary[k1] = {}
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    try:
                        json.dumps({k2: v2})
                        serializable_summary[k1][k2] = v2
                    except:
                        serializable_summary[k1][k2] = str(v2)
            else:
                serializable_summary[k1] = str(v1)
        
        json.dump(serializable_summary, f, indent=2)
    
    logging.info(f"All validation tests completed. Results saved to {output_dir}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation methods for ILS vs Introgression")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline script")
    parser.add_argument("--output", default="validation_results", help="Output directory")
    parser.add_argument("--test", choices=["recovery", "power", "robustness", "all"], 
                        default="all", help="Test to run")
    args = parser.parse_args()
    
    if args.test == "recovery":
        recovery_test = ParameterRecoveryTest(args.pipeline, args.output)
        results = recovery_test.test_introgression_parameter_recovery()
        recovery_test.analyze_recovery_results(results)
    elif args.test == "power":
        power_test = StatisticalPowerTest(args.pipeline, args.output)
        results = power_test.run_power_test()
        power_test.analyze_power_results(results)
    elif args.test == "robustness":
        robustness_test = RobustnessTest(args.pipeline, args.output)
        sample_results = robustness_test.test_sample_size_robustness()
        param_results = robustness_test.test_parameter_robustness()
        robustness_test.analyze_robustness_results(sample_results, param_results)
    else:
        run_all_validation_tests(args.pipeline, args.output)                      
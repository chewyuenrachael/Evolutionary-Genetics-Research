#!/usr/bin/env python3
"""
Statistical Analysis Module for Evolutionary Genetics
Supporting introgression vs ILS differentiation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class IntrogressionAnalyzer:
    """
    Class for statistical analysis of introgression vs ILS
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], 
                output_dir: str = "analysis_results"):
        """
        Initialize analyzer with data
        
        Parameters:
        -----------
        data : Union[str, pd.DataFrame]
            DataFrame or path to CSV file with simulation results
        output_dir : str
            Directory to save analysis results
        """
        # Load data
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Basic data cleanup
        self._clean_data()
        
    def _clean_data(self) -> None:
        """Clean and prepare data for analysis"""
        # Drop rows with errors
        if 'error' in self.df.columns:
            self.df = self.df[self.df['error'].isna()]
        
        # Convert scenarios to categorical
        if 'scenario' in self.df.columns:
            self.df['scenario'] = pd.Categorical(self.df['scenario'])
        
        # Log-transform rate variables
        for col in self.df.columns:
            if 'rate' in col.lower() and self.df[col].min() > 0:
                self.df[f'log_{col}'] = np.log10(self.df[col])
        
        # Create binary indicator for introgression presence
        if 'scenario' in self.df.columns:
            # Continuous migration is a different type of gene flow
            self.df['has_introgression'] = ~self.df['scenario'].isin(['continuous_migration'])
        
        logging.info(f"Data cleaned: {len(self.df)} valid simulations")
    
    def compute_summary_stats(self) -> pd.DataFrame:
        """
        Compute summary statistics grouped by scenario
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        if 'scenario' not in self.df.columns:
            return self.df.describe()
        
        # List of numeric columns to analyze
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['job_id', 'seed', 'length', 'has_introgression']
        analyze_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create summary by scenario
        summary = self.df.groupby('scenario')[analyze_cols].describe()
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary_stats.csv")
        summary.to_csv(summary_file)
        logging.info(f"Summary statistics saved to {summary_file}")
        
        return summary
    
    def hypothesis_testing(self) -> Dict:
        """
        Perform hypothesis tests to compare metrics across scenarios
        
        Returns:
        --------
        Dict
            Dictionary of test results
        """
        results = {}
        
        # Check if we have scenarios
        if 'scenario' not in self.df.columns:
            logging.warning("No scenario column found for hypothesis testing")
            return results
        
        # Metrics to test
        test_metrics = ['fst', 'd_stat', 'mean_internal_branch', 
                       'topology_concordance', 'private_alleles_A']
        
        # Available metrics that exist in data
        available_metrics = [m for m in test_metrics if m in self.df.columns]
        
        # Get unique scenarios
        scenarios = self.df['scenario'].unique()
        
        # Perform pairwise tests between scenarios
        for metric in available_metrics:
            metric_results = {}
            
            for i, scenario1 in enumerate(scenarios):
                for scenario2 in scenarios[i+1:]:
                    group1 = self.df[self.df['scenario'] == scenario1][metric].dropna()
                    group2 = self.df[self.df['scenario'] == scenario2][metric].dropna()
                    
                    if len(group1) < 5 or len(group2) < 5:
                        continue
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(group1, group2)
                    
                    # Effect size (Cohen's d)
                    d = (np.mean(group1) - np.mean(group2)) / \
                        np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
                    
                    metric_results[f"{scenario1}_vs_{scenario2}"] = {
                        't_stat': float(t_stat),
                        'p_value': float(p_value),
                        'u_stat': float(u_stat),
                        'u_p_value': float(u_p_value),
                        'cohens_d': float(d),
                        'mean_diff': float(np.mean(group1) - np.mean(group2)),
                        'significant': p_value < 0.05 and u_p_value < 0.05
                    }
            
            results[metric] = metric_results
        
        # Save results
        with open(os.path.join(self.output_dir, "hypothesis_tests.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Hypothesis testing completed for {len(available_metrics)} metrics")
        return results
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlations between variables
        
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        # Select numeric columns, excluding auxiliary columns
        exclude_cols = ['job_id', 'seed', 'length', 'has_introgression']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.df[corr_cols].corr()
        
        # Save correlation matrix
        corr_file = os.path.join(self.output_dir, "correlations.csv")
        corr_matrix.to_csv(corr_file)
        
        # Find strongest correlations
        corr_values = corr_matrix.unstack().sort_values(ascending=False)
        strong_corr = corr_values[corr_values < 1.0]  # Exclude self-correlations
        top_corr = strong_corr.head(20)
        
        # Save top correlations
        top_corr.to_csv(os.path.join(self.output_dir, "top_correlations.csv"))
        
        logging.info(f"Correlation analysis completed")
        return corr_matrix
    
    def pca_analysis(self) -> Dict:
        """
        Perform Principal Component Analysis on metrics
        
        Returns:
        --------
        Dict
            PCA results
        """
        # Select features for PCA
        metric_cols = ['fst', 'd_stat', 'mean_internal_branch', 'private_alleles_A', 
                      'private_alleles_B', 'shared_alleles', 'topology_concordance']
        
        # Find available metrics
        available_metrics = [col for col in metric_cols if col in self.df.columns]
        
        if len(available_metrics) < 3:
            logging.warning("Not enough metrics available for PCA")
            return {}
        
        # Select rows with no missing values
        pca_df = self.df[available_metrics].dropna()
        
        if len(pca_df) < 10:
            logging.warning("Not enough complete data points for PCA")
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_df)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result[:, :3],
            columns=['PC1', 'PC2', 'PC3']
        )
        
        # If scenario information is available
        if 'scenario' in self.df.columns:
            pca_df['scenario'] = self.df.loc[pca_df.index, 'scenario'].values
        
        # Save PCA results
        pca_df.to_csv(os.path.join(self.output_dir, "pca_results.csv"))
        
        # Save component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=available_metrics
        )
        loadings.to_csv(os.path.join(self.output_dir, "pca_loadings.csv"))
        
        # Create results dictionary
        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'loadings': loadings.to_dict(),
            'n_components': pca.n_components_
        }
        
        logging.info(f"PCA analysis completed with {pca.n_components_} components")
        return results
    
    def cluster_analysis(self) -> Dict:
        """
        Perform cluster analysis to group simulations
        
        Returns:
        --------
        Dict
            Clustering results
        """
        # Select features for clustering
        metric_cols = ['fst', 'd_stat', 'mean_internal_branch', 'topology_concordance']
        
        # Find available metrics
        available_metrics = [col for col in metric_cols if col in self.df.columns]
        
        if len(available_metrics) < 2:
            logging.warning("Not enough metrics available for clustering")
            return {}
        
        # Select rows with no missing values
        cluster_df = self.df[available_metrics].dropna()
        
        if len(cluster_df) < 10:
            logging.warning("Not enough complete data points for clustering")
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        max_clusters = min(10, len(cluster_df) // 5)
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = metrics.silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Select optimal number of clusters
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started at 2
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original dataframe
        cluster_results = cluster_df.copy()
        cluster_results['cluster'] = cluster_labels
        
        # If scenario information is available
        if 'scenario' in self.df.columns:
            cluster_results['scenario'] = self.df.loc[cluster_results.index, 'scenario'].values
        
        # Calculate cluster statistics
        cluster_stats = cluster_results.groupby('cluster').agg({
            'fst': ['mean', 'std', 'min', 'max'],
            'd_stat': ['mean', 'std', 'min', 'max']
        })
        
        # If scenario information is available, calculate scenario distribution per cluster
        if 'scenario' in cluster_results.columns:
            scenario_dist = {}
            for cluster in range(optimal_clusters):
                cluster_data = cluster_results[cluster_results['cluster'] == cluster]
                scenario_dist[f'cluster_{cluster}'] = cluster_data['scenario'].value_counts(normalize=True).to_dict()
        else:
            scenario_dist = {}
        
        # Save cluster results
        cluster_results.to_csv(os.path.join(self.output_dir, "cluster_assignments.csv"))
        
        # Create results dictionary
        results = {
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'scenario_distribution': scenario_dist,
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
        
        logging.info(f"Cluster analysis completed with {optimal_clusters} clusters")
        return results
    
    def classification_analysis(self) -> Dict:
        """
        Build and evaluate classification models for introgression detection
        
        Returns:
        --------
        Dict
            Classification results
        """
        # Check if we have the introgression indicator
        if 'has_introgression' not in self.df.columns:
            logging.warning("No introgression indicator for classification analysis")
            return {}
        
        # Select features for classification
        feature_cols = ['fst', 'd_stat', 'mean_internal_branch', 'topology_concordance',
                      'private_alleles_A', 'private_alleles_B', 'shared_alleles']
        
        # Find available features
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) < 2:
            logging.warning("Not enough features available for classification")
            return {}
        
        # Select rows with no missing values
        classification_df = self.df[available_features + ['has_introgression']].dropna()
        
        if len(classification_df) < 20:
            logging.warning("Not enough complete data points for classification")
            return {}
        
        # Split features and target
        X = classification_df[available_features]
        y = classification_df['has_introgression']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        accuracy = np.mean(y_pred == y_test)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        feature_importance.to_csv(os.path.join(self.output_dir, "feature_importance.csv"), index=False)
        
        # Create results dictionary
        results = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'feature_importance': feature_importance.to_dict(orient='records')
        }
        
        logging.info(f"Classification analysis completed, accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
        return results
    
    def pathway_analysis(self) -> Dict:
        """
        Analyze evolutionary pathways by examining parameter combinations
        
        Returns:
        --------
        Dict
            Parameter pathway analysis results
        """
        # Define parameter groups
        demographic_params = ['introgression_time', 'introgression_proportion', 'migration_rate']
        genetic_params = ['recombination_rate', 'mutation_rate']
        
        # Find available parameters
        available_demographic = [p for p in demographic_params if p in self.df.columns]
        available_genetic = [p for p in genetic_params if p in self.df.columns]
        
        results = {}
        
        # Analyze combinations of demographic and genetic parameters
        for demo_param in available_demographic:
            for genetic_param in available_genetic:
                if demo_param in self.df.columns and genetic_param in self.df.columns:
                    # Group data into bins for analysis
                    demo_bins = pd.qcut(self.df[demo_param], 4, duplicates='drop')
                    genetic_bins = pd.qcut(self.df[genetic_param], 4, duplicates='drop')
                    
                    # Calculate median FST and D-statistic for each combination
                    pathway_analysis = self.df.groupby([demo_bins, genetic_bins]).agg({
                        'fst': 'median',
                        'd_stat': 'median',
                        'mean_internal_branch': 'median'
                    }).reset_index()
                    
                    # Convert to dictionary for JSON export
                    pathway_dict = pathway_analysis.to_dict(orient='records')
                    
                    # Store results
                    results[f"{demo_param}_vs_{genetic_param}"] = pathway_dict
        
        # Save results
        with open(os.path.join(self.output_dir, "pathway_analysis.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Pathway analysis completed")
        return results
    
    def regression_analysis(self) -> Dict:
        """
        Perform regression analysis to quantify parameter effects
        
        Returns:
        --------
        Dict
            Regression results
        """
        # Define target variables
        targets = ['fst', 'd_stat', 'mean_internal_branch', 'topology_concordance']
        
        # Define predictor variables
        predictors = ['recombination_rate', 'mutation_rate', 
                     'introgression_time', 'introgression_proportion']
        
        # Find available variables
        available_targets = [t for t in targets if t in self.df.columns]
        available_predictors = [p for p in predictors if p in self.df.columns]
        
        results = {}
        
        for target in available_targets:
            target_results = {}
            
            # Select data with no missing values
            regression_df = self.df[[target] + available_predictors].dropna()
            
            if len(regression_df) < 20:
                continue
                
            # Add log-transformed predictors if they don't exist
            for pred in available_predictors:
                if 'rate' in pred and f'log_{pred}' not in regression_df.columns:
                    regression_df[f'log_{pred}'] = np.log10(regression_df[pred])
            
            # Linear regression with original scales
            y = regression_df[target]
            X = regression_df[available_predictors]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            target_results['linear'] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'significant_predictors': [pred for pred in available_predictors 
                                          if model.pvalues[pred] < 0.05]
            }
            
            # Log-transformed regression for rate variables
            log_predictors = [f'log_{p}' for p in available_predictors if 'rate' in p]
            non_log_predictors = [p for p in available_predictors if 'rate' not in p]
            log_pred_cols = log_predictors + non_log_predictors
            
            if log_predictors:
                X_log = regression_df[log_pred_cols]
                X_log = sm.add_constant(X_log)
                
                model_log = sm.OLS(y, X_log).fit()
                target_results['log_transformed'] = {
                    'r_squared': model_log.rsquared,
                    'adj_r_squared': model_log.rsquared_adj,
                    'coefficients': model_log.params.to_dict(),
                    'p_values': model_log.pvalues.to_dict(),
                    'significant_predictors': [pred for pred in log_pred_cols 
                                              if model_log.pvalues[pred] < 0.05]
                }
            
            results[target] = target_results
        
        # Save results
        with open(os.path.join(self.output_dir, "regression_analysis.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Regression analysis completed for {len(available_targets)} targets")
        return results
    
    def run_all_analyses(self) -> Dict:
        """
        Run all analysis methods and return combined results
        
        Returns:
        --------
        Dict
            Combined analysis results
        """
        results = {}
        
        # Run all analyses
        results['summary_stats'] = self.compute_summary_stats()
        results['hypothesis_tests'] = self.hypothesis_testing()
        results['correlations'] = self.correlation_analysis()
        results['pca'] = self.pca_analysis()
        results['clusters'] = self.cluster_analysis()
        results['classification'] = self.classification_analysis()
        results['pathway_analysis'] = self.pathway_analysis()
        results['regression'] = self.regression_analysis()
        
        # Save overall results
        with open(os.path.join(self.output_dir, "all_analyses.json"), 'w') as f:
            # Convert any non-serializable objects to strings or lists
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = "DataFrame saved separately"
                else:
                    try:
                        # Test if it's JSON serializable
                        json.dumps(value)
                        serializable_results[key] = value
                    except:
                        serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"All analyses completed and saved to {self.output_dir}")
        return results

def create_machine_learning_dataset(df: pd.DataFrame, 
                                  output_file: str = "ml_dataset.csv") -> pd.DataFrame:
    """
    Prepare dataset for machine learning applications
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation results DataFrame
    output_file : str
        Path to save prepared dataset
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset
    """
    # Select relevant features
    feature_cols = ['fst', 'd_stat', 'mean_internal_branch', 'topology_concordance',
                   'private_alleles_A', 'private_alleles_B', 'shared_alleles',
                   'recombination_rate', 'mutation_rate', 'scenario']
    
    # Find available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Create copy to avoid modifying original
    ml_df = df[available_features].copy()
    
    # Create binary target if scenario information is available
    if 'scenario' in ml_df.columns:
        ml_df['has_introgression'] = ~ml_df['scenario'].isin(['continuous_migration'])
        ml_df['scenario_code'] = ml_df['scenario'].astype('category').cat.codes
    
    # Log-transform rate variables
    for col in ml_df.columns:
        if 'rate' in col and ml_df[col].min() > 0:
            ml_df[f'log_{col}'] = np.log10(ml_df[col])
    
    # Fill missing values for ML (optional, depends on algorithm)
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
    ml_df[numeric_cols] = ml_df[numeric_cols].fillna(ml_df[numeric_cols].median())
    
    # Save to file
    ml_df.to_csv(output_file, index=False)
    logging.info(f"Machine learning dataset created with {len(ml_df)} samples and saved to {output_file}")
    
    return ml_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Statistical analysis for ILS vs Introgression")
    parser.add_argument("--data", required=True, help="Path to results CSV file")
    parser.add_argument("--output", default="analysis_results", help="Output directory")
    parser.add_argument("--ml", action="store_true", help="Create ML-ready dataset")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Run analysis
    analyzer = IntrogressionAnalyzer(df, args.output)
    analyzer.run_all_analyses()
    
    # Create ML dataset if requested
    if args.ml:
        create_machine_learning_dataset(df, os.path.join(args.output, "ml_dataset.csv"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy import stats
import diptest
from statsmodels.stats.multitest import multipletests
import pymc as pm
import arviz as az
from scipy.stats import probplot

def enhanced_distribution_fitting(scenario_data):
    """
    Bayesian distribution fitting using PyMC3 for a Weibull distribution.
    Returns the trace from the Bayesian model.
    """
    with pm.Model() as model:
        # Priors for Weibull parameters
        alpha = pm.HalfNormal('alpha', sigma=2)
        beta = pm.HalfNormal('beta', sigma=2)
        y_obs = pm.Weibull('y_obs', alpha=alpha, beta=beta, observed=scenario_data)
        trace = pm.sample(1000, tune=1000, cores=2, return_inferencedata=True)
    return trace

def bootstrap_multimodality(scenario_data, n_boot=100, dip_alpha=0.05):
    """
    Perform bootstrap resampling to assess multimodality.
    For each bootstrap replicate, compute:
      - Hartigan's Dip test p-value
      - Optimal number of GMM components (using BIC)
    Returns the proportion of replicates that show significant multimodality.
    """
    dip_sig_count = 0
    gmm_multi_count = 0
    n = len(scenario_data)
    for i in range(n_boot):
        sample = np.random.choice(scenario_data, size=n, replace=True)
        # Dip test
        dip_val, p_val = diptest.diptest(sample)
        if p_val < dip_alpha:
            dip_sig_count += 1
        # GMM: test 1 to 2 components (you could extend to more if desired)
        sample_reshaped = sample.reshape(-1, 1)
        bic_vals = []
        for k in [1, 2]:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(sample_reshaped)
            bic_vals.append(gmm.bic(sample_reshaped))
        optimal_components = 1 if bic_vals[0] < bic_vals[1] else 2
        if optimal_components > 1:
            gmm_multi_count += 1
    return dip_sig_count / n_boot, gmm_multi_count / n_boot

def compute_density_confidence_band(data, n_boot=100, x_grid=None):
    """
    Bootstrap the KDE (using Gaussian KDE) to compute a 95% confidence band
    for the density estimate.
    """
    if x_grid is None:
        x_grid = np.linspace(np.min(data), np.max(data), 1000)
    density_boot = []
    n = len(data)
    for i in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        kde = stats.gaussian_kde(sample)
        density_boot.append(kde.evaluate(x_grid))
    density_boot = np.array(density_boot)
    density_mean = np.mean(density_boot, axis=0)
    lower = np.percentile(density_boot, 2.5, axis=0)
    upper = np.percentile(density_boot, 97.5, axis=0)
    return x_grid, density_mean, lower, upper

def qq_plot_empirical_vs_theoretical(data, distribution, params, scenario, output_dir):
    """
    Create a Q-Q plot comparing empirical data to the theoretical quantiles.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    # Compute theoretical quantiles from the given distribution
    theoretical_quantiles = distribution.ppf(np.linspace(0.01, 0.99, len(data)), *params)
    # Empirical quantiles (sorted data)
    empirical_quantiles = np.sort(data)
    plt.plot(theoretical_quantiles, empirical_quantiles, 'o', label='Data')
    # Reference line: y = x
    min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
    max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical Quantiles')
    # Use distribution.name (e.g. "gamma", "lognorm", etc.)
    plt.title(f'Q-Q Plot for {scenario} ({distribution.name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qq_{scenario}_{distribution.name}.png", dpi=300)
    plt.close()


def analyze_branch_length_distributions(df, output_dir="analysis_output", n_boot=100):
    """
    Enhanced analysis on branch length distributions.
    
    Includes:
     - Fitting candidate distributions: exponential, gamma, lognormal, weibull.
     - Bootstrapping the dip test and GMM fitting for multimodality.
     - Plotting violin plots, density plots with 95% confidence bands,
       and Q-Q plots for the best-fit distributions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation results (must include 'mean_internal_branch' and 'scenario').
    output_dir : str
        Directory to save output figures.
    n_boot : int
        Number of bootstrap replicates to run.
    
    Returns:
    --------
    dict containing distribution fits and multimodality results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out rows with missing branch length data
    filtered_df = df.dropna(subset=['mean_internal_branch'])
    scenarios = filtered_df['scenario'].unique()
    
    # Candidate distributions to test
    dist_candidates = [
        ('exponential', stats.expon),
        ('gamma', stats.gamma),
        ('lognormal', stats.lognorm),
        ('weibull', stats.weibull_min)
    ]
    
    all_fit_results = []
    
    # Plot density fits for each scenario (with confidence bands)
    plt.figure(figsize=(15, 10))
    for i, scenario in enumerate(scenarios):
        scenario_data = filtered_df[filtered_df['scenario'] == scenario]['mean_internal_branch'].values
        if len(scenario_data) < 10:
            continue
        
        plt.subplot(2, 2, i+1)
        sns.histplot(scenario_data, stat="density", color="lightgray", label="Data", alpha=0.6)
        
        # Compute KDE confidence band
        x_grid, density_mean, lower, upper = compute_density_confidence_band(scenario_data, n_boot=n_boot)
        plt.plot(x_grid, density_mean, 'k-', label="KDE Mean")
        plt.fill_between(x_grid, lower, upper, color='gray', alpha=0.3, label="95% CI")
        
        # Fit each candidate distribution and plot its PDF
        for dist_name, distribution in dist_candidates:
            params = distribution.fit(scenario_data)
            
            # Explicitly unpack params depending on the distribution
            if dist_name == 'exponential':
                # Typically returns (loc, scale)
                loc, scale = params
                pdf_vals = distribution.pdf(x_grid, loc=loc, scale=scale)
            elif dist_name in ['gamma', 'lognormal', 'weibull']:
                # Usually returns (shape, loc, scale)
                shape, loc, scale = params
                pdf_vals = distribution.pdf(x_grid, shape, loc=loc, scale=scale)
            else:
                # Default fallback (if you add more distributions)
                pdf_vals = distribution.pdf(x_grid, *params)
            
            # Then compute log-likelihood, AIC, etc.
            log_likelihood = np.sum(distribution.logpdf(scenario_data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            
            # Plot
            plt.plot(x_grid, pdf_vals, label=f"{dist_name.capitalize()} (AIC={aic:.1f})")
            # Save Q-Q plot for best-fit later (we’ll decide best-fit based on AIC)
            all_fit_results.append({
                'scenario': scenario,
                'distribution': dist_name,
                'aic': aic,
                'params': params,
                'data': scenario_data,
                'kde_x': x_grid
            })
        plt.title(f"Density Fit: {scenario}")
        plt.xlabel("Mean Internal Branch Length")
        plt.ylabel("Density")
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/branch_length_density_fits.png", dpi=300)
    plt.close()
    
    # Convert fit results to DataFrame and find best fit per scenario (lowest AIC)
    fit_df = pd.DataFrame(all_fit_results)
    best_fits = fit_df.loc[fit_df.groupby('scenario')['aic'].idxmin()]
    print("Best-fit distributions by scenario:")
    print(best_fits[['scenario', 'distribution', 'aic']])
    
    # For each scenario, produce Q-Q plots for the best-fit distribution
    for idx, row in best_fits.iterrows():
        scenario = row['scenario']
        best_dist_name = row['distribution']
        best_dist = [d for d in dist_candidates if d[0] == best_dist_name][0][1]
        params = row['params']
        data = row['data']
        qq_plot_empirical_vs_theoretical(data, best_dist, params, scenario, output_dir)
    
    # Multiple testing correction for all KS p-values (across all fits)
    ks_results = []
    for res in all_fit_results:
        data = res['data']
        distribution = [d for d in dist_candidates if d[0] == res['distribution']][0][1]
        ks_stat, ks_pval = stats.kstest(data, distribution.cdf, args=res['params'])
        res.update({'ks_stat': ks_stat, 'ks_pval': ks_pval})
        ks_results.append(ks_pval)
    _, ks_pvals_corrected, _, _ = multipletests(ks_results, method='fdr_bh')
    
    # Add corrected p-values to fit_df
    fit_df['ks_pval'] = ks_results
    fit_df['ks_pval_corrected'] = ks_pvals_corrected
    
    # Bootstrapping multimodality detection for each scenario
    bootstrap_results = []
    for scenario in scenarios:
        scenario_data = filtered_df[filtered_df['scenario'] == scenario]['mean_internal_branch'].values
        if len(scenario_data) < 10:
            continue
        dip_boot_prop, gmm_boot_prop = bootstrap_multimodality(scenario_data, n_boot=n_boot)
        bootstrap_results.append({
            'scenario': scenario,
            'dip_bootstrap_prop': dip_boot_prop,
            'gmm_bootstrap_prop': gmm_boot_prop
        })
    bootstrap_df = pd.DataFrame(bootstrap_results)
    print("\nBootstrap multimodality results (proportion of replicates indicating multimodality):")
    print(bootstrap_df)
    
    # Bimodality detection: Original Dip test and GMM for each scenario, with plots.
    plt.figure(figsize=(15, 10))
    bimodality_results = []
    for i, scenario in enumerate(scenarios):
        scenario_data = filtered_df[filtered_df['scenario'] == scenario]['mean_internal_branch']
        if len(scenario_data) < 10:
            continue
        plt.subplot(2, 2, i+1)
        x_vals = scenario_data.values.reshape(-1, 1)
        # Evaluate GMM with 1 to 4 components
        bics = []
        n_components_range = range(1, 5)
        for k in n_components_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(x_vals)
            bics.append(gmm.bic(x_vals))
        optimal_n_components = n_components_range[np.argmin(bics)]
        gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        gmm.fit(x_vals)
        dip_val, dip_pval = diptest.diptest(scenario_data)
        bimodality_results.append({
            'scenario': scenario,
            'optimal_components': optimal_n_components,
            'dip_statistic': dip_val,
            'dip_pvalue': dip_pval
        })
        # Plot histogram and GMM fit
        sns.histplot(scenario_data, kde=True, stat="density", color="gray", alpha=0.5)
        x_plot = np.linspace(min(scenario_data), max(scenario_data), 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x_plot)
        pdf_gmm = np.exp(logprob)
        plt.plot(x_plot, pdf_gmm, '-k', label=f'GMM ({optimal_n_components} comp)')
        for j, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            comp_pdf = stats.norm.pdf(x_plot, mean[0], np.sqrt(covar[0]))
            plt.plot(x_plot, comp_pdf * gmm.weights_[j], '--', label=f'Comp {j+1} (w={gmm.weights_[j]:.2f})')
        plt.title(f"{scenario}\nDip p={dip_pval:.4f}")
        plt.xlabel("Mean Internal Branch Length")
        plt.ylabel("Density")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/branch_length_bimodality.png", dpi=300)
    
    bimodality_df = pd.DataFrame(bimodality_results)
    print("\nBimodality test results:")
    print(bimodality_df)
    
    # Enhanced visualization: Violin plot comparing branch length distributions
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='scenario', y='mean_internal_branch', data=filtered_df, inner='quartile')
    plt.title("Branch Length Comparison Across Scenarios")
    plt.ylabel("Mean Internal Branch Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/branch_length_violin.png", dpi=300)
    plt.close()
    
    # Bayesian Weibull fitting for each scenario and saving trace plots
    for scenario in scenarios:
        scenario_data = filtered_df[filtered_df['scenario'] == scenario]['mean_internal_branch'].values
        if len(scenario_data) < 10:
            continue
        try:
            trace = enhanced_distribution_fitting(scenario_data)
            az.plot_trace(trace)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/weibull_trace_{scenario}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Bayesian analysis failed for {scenario}: {str(e)}")
    
    return {
        'fit_results': fit_df,
        'best_fits': best_fits,
        'bootstrap_multimodality': bootstrap_df,
        'bimodality': bimodality_df
    }

# Example usage:
# df = pd.read_csv('simulation_results.csv')
# results = analyze_branch_length_distributions(df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon, kstest
from statsmodels.stats.multitest import multipletests
from sklearn.mixture import GaussianMixture
import diptest

def bootstrap_multimodality(data, n_boot=100, dip_alpha=0.05):
    n = len(data)
    dip_sig_count = 0
    gmm_multi_count = 0
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        dip_val, p_val = diptest.diptest(sample)
        if p_val < dip_alpha:
            dip_sig_count += 1
        sample_reshaped = sample.reshape(-1, 1)
        bic_vals = []
        for k in [1, 2]:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(sample_reshaped)
            bic_vals.append(gmm.bic(sample_reshaped))
        if bic_vals[1] < bic_vals[0]:
            gmm_multi_count += 1
    return dip_sig_count / n_boot, gmm_multi_count / n_boot

def simulate_branch_lengths(intro_frac, n=1000, scale=1.0):
    from scipy.stats import expon, norm
    n_intro = int(n * intro_frac)
    n_ils = n - n_intro
    ils = expon.rvs(scale=scale, size=n_ils)
    intro = norm.rvs(loc=0.1, scale=0.05, size=n_intro)
    intro = np.clip(intro, 0, None)
    return np.concatenate([ils, intro])

def sensitivity_analysis(intro_fracs, n_sim=50, n_points=1000):
    results = []
    for frac in intro_fracs:
        aic_vals = []
        ks_pvals = []
        bimodal_props = []
        for _ in range(n_sim):
            data = simulate_branch_lengths(frac, n=n_points, scale=1.0)
            # Fit exponential
            loc, scale = expon.fit(data)
            logL_exp = np.sum(expon.logpdf(data, loc=loc, scale=scale))
            k_exp = 2  # loc, scale
            aic_exp = 2 * k_exp - 2 * logL_exp
            aic_vals.append(aic_exp)
            # K-S test for exponential
            ks_stat, ks_pval = kstest(data, expon.cdf, args=(loc, scale))
            ks_pvals.append(ks_pval)
            # Bimodality
            dip_prop, gmm_prop = bootstrap_multimodality(data, n_boot=30)
            # average the two measures
            bimodal_props.append((dip_prop + gmm_prop) / 2)
        results.append({
            "introgression_fraction": frac,
            "mean_AIC_exponential": np.mean(aic_vals),
            "std_AIC_exponential": np.std(aic_vals),
            "mean_KS_pvalue": np.mean(ks_pvals),
            "std_KS_pvalue": np.std(ks_pvals),
            "mean_bimodality": np.mean(bimodal_props),
            "std_bimodality": np.std(bimodal_props)
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    intro_fracs = [0.0, 0.05, 0.1, 0.15, 0.2]
    df_results = sensitivity_analysis(intro_fracs, n_sim=50, n_points=1000)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    axs[0].errorbar(df_results["introgression_fraction"], 
                    df_results["mean_AIC_exponential"], 
                    yerr=df_results["std_AIC_exponential"], 
                    fmt='o-', capsize=5)
    axs[0].set_ylabel("Mean AIC (Exponential)")
    axs[0].set_title("Sensitivity Analysis: AIC vs. Introgression Fraction")

    axs[1].errorbar(df_results["introgression_fraction"], 
                    df_results["mean_KS_pvalue"], 
                    yerr=df_results["std_KS_pvalue"], 
                    fmt='s-', capsize=5, color='orange')
    axs[1].set_ylabel("Mean KS p-value")
    axs[1].set_title("Sensitivity Analysis: KS Test p-value vs. Introgression Fraction")

    axs[2].errorbar(df_results["introgression_fraction"], 
                    df_results["mean_bimodality"], 
                    yerr=df_results["std_bimodality"], 
                    fmt='d-', capsize=5, color='green')
    axs[2].set_ylabel("Mean Bimodality Proportion")
    axs[2].set_xlabel("Simulated Introgression Fraction")
    axs[2].set_title("Sensitivity Analysis: Bimodality Proportion vs. Introgression Fraction")

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png", dpi=300)
    plt.show()  # Ensure the figure is displayed

    print("Sensitivity Analysis Results:")
    print(df_results)

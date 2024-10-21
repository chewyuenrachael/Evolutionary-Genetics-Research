# Visualizing Comparison Results
# This script creates visualizations to compare simulated and empirical data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(simulated_file, empirical_file, output_file):
    """
    Plot correlation between simulated and empirical features.

    Args:
        simulated_file (str): Path to the simulated features CSV.
        empirical_file (str): Path to the empirical features CSV.
        output_file (str): Path to save the correlation plot.
    """
    # Load data
    df_sim = pd.read_csv(simulated_file)
    df_emp = pd.read_csv(empirical_file)

    # Ensure matching windows (this may require adjusting based on data)
    df_sim = df_sim.sort_values(['window_start'])
    df_emp = df_emp.sort_values(['window_start'])
    df_combined = pd.merge(df_sim, df_emp, on=['window_start', 'window_end'], suffixes=('_sim', '_emp'))

    # Plot scatter plots for each feature
    features = ['pi_Population1', 'pi_Population2', 'fst']
    for feature in features:
        plt.figure()
        plt.scatter(df_combined[f'{feature}_sim'], df_combined[f'{feature}_emp'], alpha=0.5)
        plt.xlabel(f'Simulated {feature}')
        plt.ylabel(f'Empirical {feature}')
        plt.title(f'Correlation of {feature}')
        plt.plot([df_combined[f'{feature}_sim'].min(), df_combined[f'{feature}_sim'].max()],
                 [df_combined[f'{feature}_sim'].min(), df_combined[f'{feature}_sim'].max()], 'r--')
        plt.savefig(f"{output_file}_{feature}.png")
        plt.show()

def main():
    simulated_file = "simulated_features.csv"
    empirical_file = "empirical_features.csv"
    output_file = "correlation_plot"

    plot_correlation(simulated_file, empirical_file, output_file)

if __name__ == "__main__":
    main()

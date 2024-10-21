# Comparing Simulated and Empirical Data
# This script compares features from simulated and empirical data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_features(simulated_file, empirical_file):
    """
    Compare features from simulated and empirical data.

    Args:
        simulated_file (str): Path to the simulated features CSV.
        empirical_file (str): Path to the empirical features CSV.
    """
    # Load data
    df_sim = pd.read_csv(simulated_file)
    df_emp = pd.read_csv(empirical_file)

    # Add data source labels
    df_sim['source'] = 'Simulated'
    df_emp['source'] = 'Empirical'

    # Combine data
    df = pd.concat([df_sim, df_emp], ignore_index=True)

    # Plot comparisons
    features_to_compare = ['pi_Population1', 'pi_Population2', 'fst']

    for feature in features_to_compare:
        plt.figure()
        sns.kdeplot(data=df, x=feature, hue='source', common_norm=False)
        plt.title(f'Comparison of {feature}')
        plt.show()

def main():
    simulated_file = "simulated_features.csv"
    empirical_file = "empirical_features.csv"

    compare_features(simulated_file, empirical_file)

if __name__ == "__main__":
    main()

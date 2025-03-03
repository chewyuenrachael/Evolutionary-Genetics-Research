# Calculating LD Decay from Simulated Data
# This script calculates LD decay over distance.

import msprime
import numpy as np
import pandas as pd
import tskit
import allel
import matplotlib.pyplot as plt

def calculate_ld_decay(ts, sample_indices, max_distance, num_bins, output_file):
    """
    Calculate LD decay over distance.

    Args:
        ts (msprime.TreeSequence): Simulated tree sequence.
        sample_indices (list): List of sample indices.
        max_distance (int): Maximum distance between SNPs to consider.
        num_bins (int): Number of bins for distances.
        output_file (str): Path to save the LD decay results.
    """
    # Convert tree sequence to genotype array
    haplotypes = ts.genotype_matrix()
    positions = ts.tables.sites.position
    geno = allel.HaplotypeArray(haplotypes.T).subset(sel0=sample_indices)
    geno = geno.to_genotypes(ploidy=2)

    # Calculate pairwise LD (r^2)
    loc_pairs = allel.pairwise_ld(geno, positions, max_distance=max_distance)

    # Bin LD values by distance
    distances = loc_pairs.distances
    r2 = loc_pairs.values

    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_indices = np.digitize(distances, bins)
    bin_means = [r2[bin_indices == i].mean() for i in range(1, len(bins))]

    # Save results
    df = pd.DataFrame({
        'distance': bins[:-1],
        'mean_r2': bin_means
    })
    df.to_csv(output_file, index=False)
    print(f"LD decay results saved to {output_file}")

    # Plot LD decay
    plt.figure()
    plt.plot(df['distance'], df['mean_r2'], marker='o')
    plt.xlabel('Distance (bp)')
    plt.ylabel('Mean r^2')
    plt.title('LD Decay')
    plt.show()

def main():
    # Load simulated tree sequence
    ts = tskit.load("simulated_data.trees")

    # Sample indices for the population of interest
    samples_A = ts.samples(population=0)

    # Parameters for LD calculation
    max_distance = 100000  # Maximum distance between SNPs
    num_bins = 50  # Number of bins

    # Calculate LD decay
    calculate_ld_decay(ts, samples_A, max_distance, num_bins, "ld_decay_results.csv")

if __name__ == "__main__":
    main()

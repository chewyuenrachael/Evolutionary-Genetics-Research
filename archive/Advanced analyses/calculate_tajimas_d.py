# Calculating Tajima's D from Simulated Data
# This script calculates Tajima's D for genomic windows.

import msprime
import numpy as np
import pandas as pd
import tskit
import allel

def calculate_tajimas_d(ts, window_size, sample_indices, output_file):
    """
    Calculate Tajima's D for genomic windows.

    Args:
        ts (msprime.TreeSequence): Simulated tree sequence.
        window_size (int): Size of the genomic window.
        sample_indices (list): List of sample indices.
        output_file (str): Path to save the results.
    """
    # Convert tree sequence to genotype array
    haplotypes = ts.genotype_matrix()
    positions = ts.tables.sites.position
    num_windows = int(ts.sequence_length // window_size)
    results = []

    geno = allel.HaplotypeArray(haplotypes.T).subset(sel0=sample_indices)

    for window_index in range(num_windows):
        start = window_index * window_size
        end = start + window_size
        window_mask = (positions >= start) & (positions < end)
        window_genotypes = geno[:, window_mask]
        window_positions = positions[window_mask]

        if window_genotypes.shape[1] == 0:
            continue

        # Calculate Tajima's D
        ac = window_genotypes.count_alleles()
        try:
            tajd = allel.tajima_d(ac)
        except:
            tajd = np.nan

        results.append({
            'window_start': start,
            'window_end': end,
            'tajimas_d': tajd
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Tajima's D results saved to {output_file}")

def main():
    # Load simulated tree sequence
    ts = tskit.load("simulated_data.trees")

    # Sample indices for the population of interest
    samples_A = ts.samples(population=0)

    # Set window size (e.g., 100,000 base pairs)
    window_size = 100000

    # Calculate Tajima's D
    calculate_tajimas_d(ts, window_size, samples_A, "tajimas_d_results.csv")

if __name__ == "__main__":
    main()

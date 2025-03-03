# Feature Extraction from Simulated Genomic Data
# This script extracts features from the simulated genomic data for machine learning.

import msprime
import numpy as np
import pandas as pd
import tskit
import allel

def extract_features(ts, window_size, sample_sets, output_file):
    """
    Extract features from the tree sequence.

    Args:
        ts (msprime.TreeSequence): Simulated tree sequence.
        window_size (int): Size of the genomic window for feature calculation.
        sample_sets (list): List of sample IDs for each population.
        output_file (str): Path to save the extracted features.
    """
    # Convert tree sequence to genotype array
    haplotypes = ts.genotype_matrix()
    positions = ts.tables.sites.position
    num_windows = int(ts.sequence_length // window_size)
    features = []

    for window_index in range(num_windows):
        start = window_index * window_size
        end = start + window_size
        window_mask = (positions >= start) & (positions < end)
        window_genotypes = haplotypes[:, window_mask]
        window_positions = positions[window_mask]

        if window_genotypes.size == 0:
            continue

        # Convert to allele count array
        geno = allel.HaplotypeArray(window_genotypes.T)
        ac = geno.count_alleles_subpops(sample_sets)

        # Calculate features
        feature_dict = {}
        feature_dict['window_start'] = start
        feature_dict['window_end'] = end

        # Calculate diversity (pi) for each population
        for pop_name, indices in sample_sets.items():
            pi = allel.sequence_diversity(window_positions, geno.take(indices, axis=1))
            feature_dict[f'pi_{pop_name}'] = pi

        # Calculate Fst between populations
        try:
            fst, _, _ = allel.weir_cockerham_fst(geno, subpops=list(sample_sets.values()))
            feature_dict['fst'] = np.nanmean(fst)
        except:
            feature_dict['fst'] = np.nan

        # Site Frequency Spectrum (SFS)
        sfs = geno.count_alleles().to_frequencies()
        feature_dict['sfs_mean'] = np.nanmean(sfs)
        feature_dict['sfs_var'] = np.nanvar(sfs)

        features.append(feature_dict)

    # Save features to CSV
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

def main():
    # Load simulated tree sequence
    ts = tskit.load("simulated_data.trees")

    # Define sample sets (populations)
    samples_A = ts.samples(population=0)
    samples_B = ts.samples(population=1)
    sample_sets = {
        'Species_A': samples_A,
        'Species_B': samples_B
    }

    # Set window size (e.g., 100,000 base pairs)
    window_size = 100000

    # Extract features
    extract_features(ts, window_size, sample_sets, "features.csv")

if __name__ == "__main__":
    main()

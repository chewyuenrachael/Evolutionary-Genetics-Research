# Processing Empirical Genomic Data
# This script processes empirical genomic data for comparison.

import pandas as pd
import allel
import numpy as np
import matplotlib.pyplot as plt

def process_vcf(vcf_file, output_file, populations):
    """
    Process VCF file and extract features.

    Args:
        vcf_file (str): Path to the VCF file.
        output_file (str): Path to save the processed data.
        populations (dict): Dictionary with population names and sample indices.
    """
    # Load VCF
    callset = allel.read_vcf(vcf_file)
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    positions = callset['variants/POS']
    samples = callset['samples']

    # Create sample indices for populations
    sample_indices = {}
    for pop_name, sample_ids in populations.items():
        indices = [np.where(samples == s)[0][0] for s in sample_ids]
        sample_indices[pop_name] = indices

    # Calculate features (similar to simulated data)
    features = []
    window_size = 100000
    num_windows = int(positions[-1] // window_size)

    for window_index in range(num_windows):
        start = window_index * window_size
        end = start + window_size
        window_mask = (positions >= start) & (positions < end)
        window_genotypes = genotypes.subset(sel0=window_mask)

        if window_genotypes.shape[0] == 0:
            continue

        feature_dict = {}
        feature_dict['window_start'] = start
        feature_dict['window_end'] = end

        # Convert to allele counts
        ac = window_genotypes.count_alleles_subpops(sample_indices)

        # Calculate diversity (pi) for each population
        for pop_name, indices in sample_indices.items():
            pi = allel.sequence_diversity(positions[window_mask], window_genotypes.take(indices, axis=1))
            feature_dict[f'pi_{pop_name}'] = pi

        # Calculate Fst between populations
        try:
            fst, _, _ = allel.weir_cockerham_fst(window_genotypes, subpops=list(sample_indices.values()))
            feature_dict['fst'] = np.nanmean(fst)
        except:
            feature_dict['fst'] = np.nan

        features.append(feature_dict)

    # Save features to CSV
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)
    print(f"Processed empirical data saved to {output_file}")

def main():
    # Path to the VCF file
    vcf_file = "empirical_data.vcf"

    # Define populations and samples
    populations = {
        'Population1': ['Sample1', 'Sample2', 'Sample3'],
        'Population2': ['Sample4', 'Sample5', 'Sample6']
    }

    # Output file
    output_file = "empirical_features.csv"

    # Process the VCF file
    process_vcf(vcf_file, output_file, populations)

if __name__ == "__main__":
    main()

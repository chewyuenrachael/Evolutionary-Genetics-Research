# Performing Genome Scans for Selection Using iHS
# This script calculates the Integrated Haplotype Score (iHS) to detect selection.

import msprime
import numpy as np
import pandas as pd
import tskit
import allel
import matplotlib.pyplot as plt

def perform_genome_scan(ts, sample_indices, output_file):
    """
    Perform genome scan using iHS.

    Args:
        ts (msprime.TreeSequence): Simulated tree sequence.
        sample_indices (list): List of sample indices.
        output_file (str): Path to save the iHS results.
    """
    # Convert tree sequence to haplotype array
    haplotypes = ts.genotype_matrix()
    positions = ts.tables.sites.position
    geno = allel.HaplotypeArray(haplotypes.T).subset(sel0=sample_indices)

    # Calculate allele counts
    ac = geno.count_alleles(max_allele=1)

    # Filter SNPs (biallelic, MAF > 0.05)
    is_biallelic = ac.max_allele() == 1
    af = ac.to_frequencies()
    maf_filter = (af[:, 1] > 0.05) & (af[:, 1] < 0.95)
    snp_filter = is_biallelic & maf_filter

    geno_filt = geno.compress(snp_filter, axis=1)
    positions_filt = positions[snp_filter]

    # Compute haplotype homozygosity (EHH)
    ehh = allel.ehh_decay(positions_filt, geno_filt)

    # Compute iHS
    ihs_scores = allel.ihs(ac[snp_filter], geno_filt, positions_filt)

    # Save results
    df = pd.DataFrame({
        'position': positions_filt,
        'iHS': ihs_scores
    })
    df.to_csv(output_file, index=False)
    print(f"iHS results saved to {output_file}")

    # Plot iHS scores
    plt.figure(figsize=(12, 6))
    plt.plot(df['position'], df['iHS'], '.', alpha=0.5)
    plt.xlabel('Genomic Position (bp)')
    plt.ylabel('iHS')
    plt.title('Genome Scan Using iHS')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='red', linestyle='--')
    plt.show()

def main():
    # Load simulated tree sequence
    ts = tskit.load("simulated_data.trees")

    # Sample indices for the population of interest
    samples_A = ts.samples(population=0)

    # Perform genome scan
    perform_genome_scan(ts, samples_A, "ihs_results.csv")

if __name__ == "__main__":
    main()

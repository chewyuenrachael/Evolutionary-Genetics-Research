# ✅ Here's the fully reviewed and finalized version of `common_utils.py` with:
# - Correct imports
# - Consistent docstrings
# - Robust structure
# - Cleaned up comments and logic

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tskit

# ------------------------------
# Save Utilities
# ------------------------------

def save_dataframe(df, filepath):
    """Save a pandas DataFrame to CSV, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

# ------------------------------
# D-Statistic Calculation (ABBA-BABA)
# ------------------------------

def calculate_d_statistic(ts):
    """
    Calculates Patterson’s D-statistic using allele frequencies
    for a tree sequence with ((A,B),C) topology.
    """
    genotype_matrix = ts.genotype_matrix()
    samples = np.array([ts.get_population(n) for n in ts.samples()])

    a_idx = np.where(samples == 0)[0]
    b_idx = np.where(samples == 1)[0]
    c_idx = np.where(samples == 2)[0]

    abba = baba = 0

    for row in genotype_matrix:
        p1 = np.mean(row[a_idx])  # Population A
        p2 = np.mean(row[b_idx])  # Population B
        p3 = np.mean(row[c_idx])  # Population C

        abba += (1 - p1) * p2 * p3
        baba += p1 * (1 - p2) * p3

    denom = abba + baba
    d_stat = (abba - baba) / denom if denom > 0 else 0.0

    return d_stat, abba, baba

# ------------------------------
# FST Calculation (Weir and Cockerham)
# ------------------------------

def calculate_fst(ts):
    """
    Pairwise FST between A-B and B-C from tree sequence.
    Uses population IDs (0 = A, 1 = B, 2 = C).
    """
    genotypes = ts.genotype_matrix()
    if genotypes.shape[0] == 0:
        return 0.0, 0.0

    samples = ts.samples()
    populations = np.array([ts.node(n).population for n in samples])

    pop_a_idx = np.where(populations == 0)[0]
    pop_b_idx = np.where(populations == 1)[0]
    pop_c_idx = np.where(populations == 2)[0]

    def pairwise_fst(idx1, idx2):
        hap1 = genotypes[:, idx1]
        hap2 = genotypes[:, idx2]

        p1 = np.mean(hap1, axis=1)
        p2 = np.mean(hap2, axis=1)

        n1, n2 = len(idx1), len(idx2)
        n_tot = n1 + n2
        p_bar = (n1 * p1 + n2 * p2) / n_tot

        h_s = (n1 * p1 * (1 - p1) + n2 * p2 * (1 - p2)) / n_tot
        h_t = p_bar * (1 - p_bar)

        valid = h_t > 0
        fst = 1 - (h_s[valid] / h_t[valid])
        fst = fst[~np.isnan(fst)]

        return np.mean(fst) if len(fst) > 0 else 0.0

    fst_ab = pairwise_fst(pop_a_idx, pop_b_idx)
    fst_bc = pairwise_fst(pop_b_idx, pop_c_idx)

    return fst_ab, fst_bc



# ------------------------------
# Publication-Ready KDE Plot
# ------------------------------

def plot_kde_distribution(
    data, group_col, value_col, title, xlabel,
    output_path=None, xlim=None, sig_lines=None
):
    """
    Publication-ready KDE plot grouped by a categorical column.
    """
    plt.figure(figsize=(10, 6))

    # Custom color palette
    color_map = {
        "ancient": "#8B0000",        # dark red
        "intermediate": "#E69500",  # orange
        "recent": "#FFD700"         # yellow
    }

    for group in sorted(data[group_col].unique()):
        subset = data[data[group_col] == group][value_col]
        sns.kdeplot(
            subset,
            label=group,
            fill=True,
            linewidth=2,
            color=color_map.get(group, None),
            alpha=0.5
        )

    plt.axvline(0, color="red", linestyle="--", label="D = 0", alpha=0.7)

    if sig_lines:
        for val in sig_lines:
            plt.axvline(val, color="grey", linestyle="--", alpha=0.6)
    
    if xlabel.lower().startswith("d") and not sig_lines:
        plt.axvline(0.2, color="grey", linestyle="--", alpha=0.6)
        plt.axvline(-0.2, color="grey", linestyle="--", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(title=group_col.capitalize())
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# ------------------------------
# Sliding Window D-statistic
# ------------------------------

def sliding_window_d(ts, window_size=10000):
    """
    Computes D-statistic in sliding windows across the genome.
    """
    genotype_matrix = ts.genotype_matrix()
    positions = np.array([var.position for var in ts.variants()])
    samples = np.array([ts.get_population(n) for n in ts.samples()])

    a_idx = np.where(samples == 0)[0]
    b_idx = np.where(samples == 1)[0]
    c_idx = np.where(samples == 2)[0]

    p1_idx = np.random.choice(a_idx)
    p2_idx = np.random.choice(b_idx)
    p3_idx = np.random.choice(c_idx)

    windows = np.arange(0, ts.sequence_length, window_size)
    if windows[-1] < ts.sequence_length:
        windows = np.append(windows, ts.sequence_length)

    mids, d_stats = [], []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        indices = (positions >= start) & (positions < end)
        sites = np.where(indices)[0]

        abba = baba = 0
        for s in sites:
            p1 = genotype_matrix[s, p1_idx]
            p2 = genotype_matrix[s, p2_idx]
            p3 = genotype_matrix[s, p3_idx]

            if p1 == 0 and p2 == 1 and p3 == 1:
                abba += 1
            elif p1 == 1 and p2 == 0 and p3 == 1:
                baba += 1

        if (abba + baba) > 0:
            d_stat = (abba - baba) / (abba + baba)
        else:
            d_stat = 0.0

        mids.append((start + end) / 2)
        d_stats.append(d_stat)

    return mids, d_stats

# ------------------------------
# Introgression Scenario Simulation
# ------------------------------

def simulate_introgression_scenario(
    Ne,
    div_time_ab,
    div_time_abc,
    mig_rate,
    mig_time,
    mig_duration=100,
    migration_direction="C_to_B",
    continuous=False,
    num_samples=(10, 10, 10),
    sequence_length=1e6,
    recombination_rate=1e-8,
    mutation_rate=1e-8
):
    import msprime

    demography = msprime.Demography()

    # Step 1: Add leaf populations
    demography.add_population(name="A", initial_size=Ne)
    demography.add_population(name="B", initial_size=Ne)
    demography.add_population(name="C", initial_size=Ne)
    
    # Step 2: Add ancestor (shared by all three)
    demography.add_population(name="Ancestor", initial_size=Ne)

    # Step 3: Perform splits (latest to earliest!)
    # Split A and B first from Ancestor
    demography.add_population_split(
        time=div_time_ab, derived=["A", "B"], ancestral="Ancestor"
    )

    # Now split C from Ancestor
    demography.add_population_split(
        time=div_time_abc, derived=["C"], ancestral="Ancestor"
    )

    # Step 4: Migration
    if mig_rate > 0:
        if continuous:
            if migration_direction == "C_to_B":
                demography.set_migration_rate("C", "B", rate=mig_rate)
            else:
                demography.set_migration_rate("B", "C", rate=mig_rate)
        else:
            src = "C" if migration_direction == "C_to_B" else "B"
            dst = "B" if migration_direction == "C_to_B" else "C"
            demography.add_mass_migration(
                time=mig_time, source=dst, dest=src, proportion=mig_rate
            )

    demography.sort_events()

    # Step 5: Sampling
    samples = [
        msprime.SampleSet(num_samples[0], population="A"),
        msprime.SampleSet(num_samples[1], population="B"),
        msprime.SampleSet(num_samples[2], population="C"),
    ]

    # Step 6: Run simulations
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=np.random.randint(1e6)
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate)

    return ts


def sliding_window_fst(ts, window_size=10000):
    """Compute FST in sliding windows between populations B and C."""
    genotypes = ts.genotype_matrix()
    positions = np.array([v.position for v in ts.variants()])
    samples = np.array([ts.get_population(n) for n in ts.samples()])
    
    b_idx = np.where(samples == 1)[0]
    c_idx = np.where(samples == 2)[0]

    windows = np.arange(0, ts.sequence_length, window_size)
    if windows[-1] < ts.sequence_length:
        windows = np.append(windows, ts.sequence_length)

    mids = []
    fst_vals = []

    for i in range(len(windows) - 1):
        start, end = windows[i], windows[i + 1]
        indices = (positions >= start) & (positions < end)
        sites = np.where(indices)[0]

        if len(sites) == 0:
            continue

        hap1 = genotypes[sites][:, b_idx]
        hap2 = genotypes[sites][:, c_idx]

        p1 = np.mean(hap1, axis=1)
        p2 = np.mean(hap2, axis=1)

        n1, n2 = len(b_idx), len(c_idx)
        p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)

        h_s = (n1 * p1 * (1 - p1) + n2 * p2 * (1 - p2)) / (n1 + n2)
        h_t = p_bar * (1 - p_bar)

        valid = h_t > 0
        if np.sum(valid) == 0:
            continue

        fst = 1 - np.mean(h_s[valid] / h_t[valid])
        fst_vals.append(fst)
        mids.append((start + end) / 2)

    return mids, fst_vals



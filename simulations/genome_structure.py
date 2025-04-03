# genome_structure.py
# Simulates introgression with realistic genome structure
# Incorporates non-uniform recombination and mutation rates

import msprime
import numpy as np
import pandas as pd
import tskit
import random
from common_utils import calculate_d_statistic, calculate_fst, plot_d_kde, plot_fst_kde

# Constants
SEQ_LENGTH = 1_000_000  # 1Mb
N_REPLICATES = 30
NUM_SAMPLES_PER_POP = 10
NE = 10_000

# Genome features: recombination/mutation hotspots
recomb_map = msprime.RecombinationMap.uniform_map(
    length=SEQ_LENGTH,
    rate=1e-8,
    num_loci=SEQ_LENGTH
)

# Modify with hotspots (example: high recomb between 400kb–500kb)
recomb_map.set_rate(4e-8, 400_000, 500_000)

# Optional: mutate map if desired
mutation_rate_map = np.full(SEQ_LENGTH, 1e-8)
mutation_rate_map[200_000:300_000] = 5e-8  # hotspot
mutation_rate_map[700_000:800_000] = 2e-9  # coldspot

# Demographic model with introgression and hotspots
def simulate_genome_structure(migration_rate=1e-2, migration_time=50_000):
    demography = msprime.Demography()
    demography.add_population("A", initial_size=NE)
    demography.add_population("B", initial_size=NE)
    demography.add_population("C", initial_size=NE)
    demography.add_population("AB", initial_size=NE)
    demography.add_population("ABC", initial_size=NE)

    # Add migration pulse from C to B
    demography.add_migration_rate_change(time=migration_time, rate=migration_rate, source="B", dest="C")
    demography.add_migration_rate_change(time=migration_time + 10_000, rate=0, source="B", dest="C")

    demography.add_population_split(time=100_000, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=300_000, derived=["AB", "C"], ancestral="ABC")

    demography.sort_events()

    # Simulate ancestry
    ts = msprime.sim_ancestry(
        samples={"A": NUM_SAMPLES_PER_POP, "B": NUM_SAMPLES_PER_POP, "C": NUM_SAMPLES_PER_POP},
        recombination_map=recomb_map,
        demography=demography,
        sequence_length=SEQ_LENGTH,
        random_seed=random.randint(1, 1_000_000)
    )

    # Mutations (site-by-site)
    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate_map,
        discrete_genome=False,
        random_seed=random.randint(1, 1_000_000)
    )

    return ts

if __name__ == '__main__':
    d_stats = []
    fst_vals = []

    print("Simulating genome with structure...")
    for _ in range(N_REPLICATES):
        ts = simulate_genome_structure()

        d = calculate_d_statistic(ts)
        fst_ab, fst_bc = calculate_fst(ts)

        d_stats.append(d)
        fst_vals.append(fst_ab)

    # Convert to DataFrame
    df = pd.DataFrame({
        "D": d_stats,
        "FST_AB": fst_vals
    })

    # Save results
    df.to_csv("results/genome_structure_results.csv", index=False)

    # Plot results
    plot_d_kde(df, label="Genome-structured D")
    plot_fst_kde(df, label="Genome-structured FST")

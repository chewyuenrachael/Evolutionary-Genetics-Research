# Enhanced script for simulating gene genealogies under a coalescent model
# with recombination, incomplete lineage sorting (ILS), and introgression events
# using msprime 1.x.

import msprime
import numpy as np
import matplotlib.pyplot as plt

def simulate_genomes(
    recombination_rate,
    mutation_rate,
    length,
    samples,
    demography,
    introgression_events=None,
    random_seed=None
):
    """
    Simulate genomes under a complex demographic model with recombination.

    Args:
        recombination_rate (float): Recombination rate per base pair per generation.
        mutation_rate (float): Mutation rate per base pair per generation.
        length (int): Length of the simulated genome in base pairs.
        samples (list): List of msprime.SampleSet objects specifying samples to be taken.
        demography (msprime.Demography): Demography object defining populations and events.
        introgression_events (list): List of dictionaries specifying introgression events.
        random_seed (int): Seed for random number generator.

    Returns:
        ts (msprime.TreeSequence): Simulated tree sequence.
    """
    if introgression_events is None:
        introgression_events = []

    # Add introgression events as mass migrations
    for event in introgression_events:
        demography.add_mass_migration(
            time=event['time'],
            source=event['recipient'],
            dest=event['donor'],
            proportion=event['proportion']
        )

    # Simulate ancestry with msprime
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        recombination_rate=recombination_rate,
        sequence_length=length,
        random_seed=random_seed
    )

    # Add mutations
    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        random_seed=random_seed
    )

    return ts

def main():
    # Simulation parameters
    demography = msprime.Demography()
    demography.add_population(name="Species_A", initial_size=10000)
    demography.add_population(name="Species_B", initial_size=10000)
    demography.add_population(name="Outgroup", initial_size=10000)

    # Species split
    demography.add_population_split(
        time=5000,
        derived=["Species_A", "Species_B"],
        ancestral="Outgroup"
    )

    # Split between Species_A and Species_B
    demography.add_population_split(
        time=2000,
        derived=["Species_B"],
        ancestral="Species_A"
    )

    # Introgression events
    introgression_events = [
        {'time': 1500, 'donor': "Species_B", 'recipient': "Species_A", 'proportion': 0.1}
    ]

    recombination_rate = 1e-8  # Per base pair per generation
    mutation_rate = 1e-8       # Per base pair per generation
    length = 1e7               # Length of the genome in base pairs

    samples = [
        msprime.SampleSet(5, population="Species_A", time=0),
        msprime.SampleSet(5, population="Species_B", time=0),
        msprime.SampleSet(5, population="Outgroup", time=0)
    ]

    # Run the simulation
    ts = simulate_genomes(
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        length=length,
        samples=samples,
        demography=demography,
        introgression_events=introgression_events,
        random_seed=42
    )

    # Analyze the results
    # Calculate Site Frequency Spectrum
    sfs = ts.allele_frequency_spectrum(mode="site", polarized=False)
    print("Site Frequency Spectrum:")
    print(sfs)

    # Plot the Site Frequency Spectrum
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sfs)), sfs, color='skyblue')
    plt.xlabel('Allele Frequency')
    plt.ylabel('Number of Sites')
    plt.title('Site Frequency Spectrum')
    plt.show()

    # Calculate FST between populations
    pop_A = ts.samples(population=demography.population("Species_A").id)
    pop_B = ts.samples(population=demography.population("Species_B").id)
    FST = ts.Fst([pop_A, pop_B])
    print(f"FST between Species_A and Species_B: {FST}")

    # Perform D-statistic analysis to distinguish introgression from ILS
    # For D-statistic, we need four taxa; add an additional outgroup
    demography.add_population(name="Outgroup2", initial_size=10000)
    demography.add_population_split(
        time=10000,
        derived=["Outgroup"],
        ancestral="Outgroup2"
    )
    samples.append(msprime.SampleSet(5, population="Outgroup2", time=0))

    # Re-run the simulation with the new outgroup
    ts = simulate_genomes(
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        length=length,
        samples=samples,
        demography=demography,
        introgression_events=introgression_events,
        random_seed=42
    )

    # Get sample indices for each population
    pop_A = ts.samples(population=demography.population("Species_A").id)
    pop_B = ts.samples(population=demography.population("Species_B").id)
    pop_O1 = ts.samples(population=demography.population("Outgroup").id)
    pop_O2 = ts.samples(population=demography.population("Outgroup2").id)

    # Convert genotype matrix
    G = ts.genotype_matrix().T  # Shape (num_samples, num_sites)

    # Prepare allele counts per population
    import allel
    haplotypes = G
    ac1 = haplotypes[:, pop_A].sum(axis=1)
    ac2 = haplotypes[:, pop_B].sum(axis=1)
    ac3 = haplotypes[:, pop_O1].sum(axis=1)
    ac4 = haplotypes[:, pop_O2].sum(axis=1)

    # Compute D-statistics using scikit-allel
    d_stat, f4 = allel.moving_patterson_d(
        ac1[:, np.newaxis],
        ac2[:, np.newaxis],
        ac3[:, np.newaxis],
        ac4[:, np.newaxis],
        size=1000, step=1000
    )

    # Plot D-statistic
    plt.figure(figsize=(10, 6))
    plt.plot(d_stat)
    plt.xlabel('Window')
    plt.ylabel('D-statistic')
    plt.title('D-statistic across genome')
    plt.show()

    # Save the tree sequence for further analysis
    ts.dump("enhanced_simulated_data.trees")

if __name__ == "__main__":
    main()

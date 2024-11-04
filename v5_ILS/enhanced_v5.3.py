# Enhanced script for simulating gene genealogies under a coalescent model
# with recombination, incomplete lineage sorting (ILS), and introgression events
# using msprime 1.x, including parameter sampling, parallelization, and advanced analysis.

import msprime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import random
import logging
import os

def simulate_genomes(
    recombination_rate,
    mutation_rate,
    length,
    samples,
    demography,
    introgression_event,
    random_seed
):
    """
    Simulate genomes under a complex demographic model with recombination.

    Args:
        recombination_rate (float): Recombination rate per base pair per generation.
        mutation_rate (float): Mutation rate per base pair per generation.
        length (int): Length of the simulated genome in base pairs.
        samples (list): List of msprime.SampleSet objects specifying samples to be taken.
        demography (msprime.Demography): Demography object defining populations and events.
        introgression_event (dict): Dictionary specifying introgression event.
        random_seed (int): Seed for random number generator.

    Returns:
        ts (msprime.TreeSequence): Simulated tree sequence.
    """
    
    # Validate recombination rate
    if recombination_rate <= 0 or recombination_rate > 1e-6:
        raise ValueError("Recombination rate must be a positive value and below 1e-6.")

    # Validate mutation rate
    if mutation_rate <= 0 or mutation_rate > 1e-6:
        raise ValueError("Mutation rate must be a positive value and below 1e-6.")

    # Validate length of genome
    if length <= 0:
        raise ValueError("Genome length must be a positive integer.")

    # Validate introgression parameters
    if not (0 < introgression_event['proportion'] <= 1):
        raise ValueError("Introgression proportion must be between 0 and 1.")
    
    # Validate population structure
    if introgression_event['donor'] not in [pop.name for pop in demography.populations] or \
       introgression_event['recipient'] not in [pop.name for pop in demography.populations]:
        raise ValueError("Introgression donor and recipient must be valid population names.")
    
    # Create a copy of the demography to avoid modifying the original
    demography_sim = demography.copy()
    
    # Add introgression event as mass migration
    demography_sim.add_mass_migration(
        time=introgression_event['time'],
        source=introgression_event['recipient'],
        dest=introgression_event['donor'],
        proportion=introgression_event['proportion']
    )

    # Simulate ancestry with msprime
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography_sim,
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

def analyze_simulation(ts, demography):
    """
    Analyze the tree sequence to compute F_ST and D-statistics.

    Args:
        ts (msprime.TreeSequence): Simulated tree sequence.
        demography (msprime.Demography): Demography object to map populations.

    Returns:
        results (dict): Dictionary containing analysis results.
    """
    # Get sample indices for each population
    pop_ids = {pop.name: pop.id for pop in demography.populations}
    samples_A = ts.samples(population=pop_ids["Species_A"])
    samples_B = ts.samples(population=pop_ids["Species_B"])
    samples_O1 = ts.samples(population=pop_ids["Outgroup"])
    samples_O2 = ts.samples(population=pop_ids["Outgroup2"])

    # Calculate F_ST between Species_A and Species_B
    FST = ts.Fst([samples_A, samples_B])

    # Compute D-statistics
    import allel
    # Extract genotype matrix
    G = ts.genotype_matrix().T  # Shape (num_samples, num_sites)

    # Prepare allele counts per population
    haplotypes = G
    ac1 = haplotypes[:, samples_A].sum(axis=1)
    ac2 = haplotypes[:, samples_B].sum(axis=1)
    ac3 = haplotypes[:, samples_O1].sum(axis=1)
    ac4 = haplotypes[:, samples_O2].sum(axis=1)

    # Filter biallelic sites
    biallelic_mask = (ac1 + ac2 + ac3 + ac4 <= 2 * len(samples_A) + 2 * len(samples_B) + 2 * len(samples_O1) + 2 * len(samples_O2))
    ac1 = ac1[biallelic_mask]
    ac2 = ac2[biallelic_mask]
    ac3 = ac3[biallelic_mask]
    ac4 = ac4[biallelic_mask]

    # Compute D-statistic
    num = np.sum((ac1 - ac2) * (ac3 - ac4))
    den = np.sum((ac1 + ac2) * (ac3 + ac4))
    D = num / den if den != 0 else np.nan

    results = {
        'FST': FST,
        'D_statistic': D
    }

    return results

def simulation_worker(sim_id, params):
    """
    Worker function to run a single simulation and analysis.

    Args:
        sim_id (int): Simulation identifier.
        params (dict): Dictionary of parameters for the simulation.

    Returns:
        results (dict): Dictionary containing simulation ID, parameters, and analysis results.
    """
    random_seed = params['random_seed']
    recombination_rate = params['recombination_rate']
    mutation_rate = params['mutation_rate']
    length = params['length']
    introgression_event = params['introgression_event']
    demography = params['demography']
    samples = params['samples']

    # Run the simulation
    ts = simulate_genomes(
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        length=length,
        samples=samples,
        demography=demography,
        introgression_event=introgression_event,
        random_seed=random_seed
    )

    # Analyze the simulation
    analysis_results = analyze_simulation(ts, demography)

    # Compile all results
    results = {
        'sim_id': sim_id,
        'random_seed': random_seed,
        'recombination_rate': recombination_rate,
        'mutation_rate': mutation_rate,
        'introgression_time': introgression_event['time'],
        'introgression_proportion': introgression_event['proportion'],
        'FST': analysis_results['FST'],
        'D_statistic': analysis_results['D_statistic']
    }

    return results

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # Simulation parameters
    num_simulations = 100

    # Define ranges for parameter sampling
    recombination_rate_range = (1e-9, 1e-8)
    mutation_rate_range = (1e-9, 1e-8)
    introgression_time_range = (1000, 3000)
    introgression_proportion_range = (0.05, 0.2)
    length = 1e7  # Genome length

    # Base demography
    demography = msprime.Demography()
    demography.add_population(name="Species_A", initial_size=10000)
    demography.add_population(name="Species_B", initial_size=10000)
    demography.add_population(name="Outgroup", initial_size=10000)
    demography.add_population(name="Outgroup2", initial_size=10000)

    # Species split events
    demography.add_population_split(
        time=10000,
        derived=["Outgroup"],
        ancestral="Outgroup2"
    )
    demography.add_population_split(
        time=5000,
        derived=["Species_A", "Species_B"],
        ancestral="Outgroup"
    )
    demography.add_population_split(
        time=2000,
        derived=["Species_B"],
        ancestral="Species_A"
    )

    # Sample sets
    samples = [
        msprime.SampleSet(5, population="Species_A", time=0),
        msprime.SampleSet(5, population="Species_B", time=0),
        msprime.SampleSet(5, population="Outgroup", time=0),
        msprime.SampleSet(5, population="Outgroup2", time=0)
    ]

    # Prepare simulation parameters
    simulation_params_list = []
    for sim_id in range(num_simulations):
        # Randomly sample parameters within specified ranges
        recombination_rate = 10 ** np.random.uniform(
            np.log10(recombination_rate_range[0]),
            np.log10(recombination_rate_range[1])
        )
        mutation_rate = 10 ** np.random.uniform(
            np.log10(mutation_rate_range[0]),
            np.log10(mutation_rate_range[1])
        )
        introgression_time = np.random.uniform(
            introgression_time_range[0],
            introgression_time_range[1]
        )
        introgression_proportion = np.random.uniform(
            introgression_proportion_range[0],
            introgression_proportion_range[1]
        )
        random_seed = random.randint(1, 1e6)

        introgression_event = {
            'time': introgression_time,
            'donor': "Species_B",
            'recipient': "Species_A",
            'proportion': introgression_proportion
        }

        params = {
            'random_seed': random_seed,
            'recombination_rate': recombination_rate,
            'mutation_rate': mutation_rate,
            'length': length,
            'introgression_event': introgression_event,
            'demography': demography,
            'samples': samples
        }
        simulation_params_list.append((sim_id, params))

    # Run simulations in parallel
    logging.info("Starting simulations...")
    results_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulation_worker, sim_id, params)
                   for sim_id, params in simulation_params_list]
        for future in futures:
            result = future.result()
            results_list.append(result)
            logging.info(f"Completed simulation {result['sim_id']}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to CSV
    results_df.to_csv("simulation_results.csv", index=False)
    logging.info("Simulation results saved to 'simulation_results.csv'.")

    # Statistical analysis
    # Calculate mean and standard deviation of FST and D-statistics
    fst_mean = results_df['FST'].mean()
    fst_std = results_df['FST'].std()
    d_stat_mean = results_df['D_statistic'].mean()
    d_stat_std = results_df['D_statistic'].std()

    logging.info(f"FST Mean: {fst_mean}, FST Std Dev: {fst_std}")
    logging.info(f"D-statistic Mean: {d_stat_mean}, D-statistic Std Dev: {d_stat_std}")

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['FST'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('F_ST')
    plt.ylabel('Frequency')
    plt.title('Distribution of F_ST across Simulations')

    plt.subplot(1, 2, 2)
    plt.hist(results_df['D_statistic'], bins=20, color='salmon', edgecolor='black')
    plt.xlabel('D-statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of D-statistics across Simulations')

    plt.tight_layout()
    plt.savefig("simulation_histograms.png")
    plt.show()
    logging.info("Histograms saved to 'simulation_histograms.png'.")

    # Additional advanced analysis and visualization can be added here

if __name__ == "__main__":
    main()

import msprime

# Set the number of samples, mutation rate, and recombination rate
n = 20  # Increase the sample size
mu = 1e-8  # Modify the mutation rate based on empirical data
rho = 1e-8  # Modify the recombination rate based on empirical data
population_size = 100  # Adjust the population size based on empirical data
seed = 42
t_divergence = 100  # Time of divergence between populations
t_admixture = 70  # Time of the admixture event
proportion_admixture = 0.2  # Proportion of population 0 migrating into population 1

# Define demographic events: sample configuration, mass migration, and bottleneck
population_configurations = [
    msprime.PopulationConfiguration(sample_size=n, initial_size=population_size),
    msprime.PopulationConfiguration(sample_size=n, initial_size=population_size),
]

# Add instantaneous bottleneck event and migration events
demographic_events = [
    msprime.PopulationParametersChange(time=20, initial_size=200, population_id=0),  # Change in population size
    msprime.InstantaneousBottleneck(time=30, population=0, strength=0.5),  # Instantaneous bottleneck event
    msprime.MassMigration(time=t_admixture, source=0, destination=1, proportion=proportion_admixture),
    msprime.MassMigration(time=t_divergence, source=0, destination=1, proportion=1.0),
]

ts = msprime.simulate(
    population_configurations=population_configurations,
    demographic_events=demographic_events,
    mutation_rate=mu,
    recombination_rate=rho,
    random_seed=seed,
)

# Print trees and branch lengths
for tree in ts.trees():
    print(tree.draw(format="unicode"))
    
    # Print branch lengths
    for u in tree.nodes():
        if tree.parent(u) != msprime.NULL_NODE:
            branch_length = tree.time(tree.parent(u)) - tree.time(u)
            print(f"Branch length of node {u} to its parent: {branch_length}")
    print("\n")

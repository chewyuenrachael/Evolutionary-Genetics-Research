
"""
Incomplete Lineage Sorting and Introgression Simulation

This script simulates gene genealogies under a coalescent model with incomplete lineage sorting (ILS)
and introgression events. It allows for the specification of parameters such as population size,
speciation times, introgression events, mutation rates, and more. The output includes the simulated gene tree
and the site frequency spectrum (SFS), which can be used for further analysis.
"""

import numpy as np

class Node:
    """
    A class to represent a node in a genealogy tree.

    Attributes:
        left (Node): Left child node.
        right (Node): Right child node.
        parent (Node): Parent node.
        age (float): Time in generations when the node was formed.
        species (str): Identifier for the species.
        label (str): Unique label for the node.
        mutations (list): List of mutations on this branch.
    """

    def __init__(self, left=None, right=None, age=0.0, species=None, label=None):
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.parent = None           # Parent node
        self.age = age               # Age of the node (time in generations)
        self.species = species       # Species identifier
        self.label = label           # Unique label for the node
        self.mutations = []          # List to store mutations on this branch

        # Set parent references in child nodes
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self

    def is_leaf(self):
        """Check if the node is a leaf (tip) node."""
        return self.left is None and self.right is None

    def get_leaf_labels(self):
        """Recursively get labels of all descendant leaf nodes."""
        if self.is_leaf():
            return [self.label]
        labels = []
        if self.left is not None:
            labels.extend(self.left.get_leaf_labels())
        if self.right is not None:
            labels.extend(self.right.get_leaf_labels())
        return labels

    def __str__(self):
        return f"Node(label={self.label}, age={self.age:.2f}, species={self.species})"

    def __repr__(self):
        return self.__str__()

def simulate_coalescent_with_introgression(rng, N_e, samples_per_species, species_split_times,
                                           introgression_events, mutation_rate):
    """
    Simulate a coalescent process with ILS and introgression events.

    Args:
        rng (Generator): NumPy random number generator.
        N_e (float): Effective population size.
        samples_per_species (dict): Number of samples per species.
        species_split_times (dict): Times when species split from each other.
        introgression_events (list): List of introgression events (dicts).
        mutation_rate (float): Mutation rate per generation.

    Returns:
        Node: Root of the coalescent tree.
    """

    # Initialize nodes for each species
    nodes = []
    for species, sample_size in samples_per_species.items():
        for i in range(sample_size):
            label = f"{species}_{i}"
            node = Node(age=0.0, species=species, label=label)
            nodes.append(node)

    active_lineages = nodes.copy()
    current_time = 0.0

    # Combine all event times
    event_times = set(species_split_times.values())
    event_times.update([event['time'] for event in introgression_events])
    event_times = sorted(event_times)

    # Main simulation loop
    for next_event_time in event_times + [np.inf]:
        while current_time < next_event_time and len(active_lineages) > 1:
            # Calculate total rate of coalescence
            k = len(active_lineages)
            if k < 2:
                break
            total_coalescence_rate = k * (k - 1) / (4 * N_e)
            # Time to next coalescent event
            wait_time = rng.exponential(1 / total_coalescence_rate)
            current_time += wait_time

            if current_time >= next_event_time:
                # Event occurs before next coalescence
                current_time = next_event_time
                break

            # Coalescence event
            # Choose two lineages from the same species
            species_lineages = {}
            for node in active_lineages:
                species_lineages.setdefault(node.species, []).append(node)

            possible_species = [species for species in species_lineages if len(species_lineages[species]) >= 2]
            if not possible_species:
                continue

            # Randomly select a species and pair to coalesce
            species = rng.choice(possible_species)
            lineage_pair = rng.choice(species_lineages[species], size=2, replace=False)
            new_node = Node(left=lineage_pair[0], right=lineage_pair[1], age=current_time,
                            species=species, label=f"Node_{current_time:.2f}")
            active_lineages.remove(lineage_pair[0])
            active_lineages.remove(lineage_pair[1])
            active_lineages.append(new_node)

        # Process events at next_event_time
        events_at_time = [event for event in introgression_events if event['time'] == current_time]
        for event in events_at_time:
            # Introgression event
            donor_species = event['donor_species']
            recipient_species = event['recipient_species']
            # Choose a donor lineage
            donor_lineages = [node for node in active_lineages if node.species == donor_species]
            if donor_lineages:
                donor_node = rng.choice(donor_lineages)
                # Clone donor node into recipient species
                introgressed_node = Node(left=donor_node.left, right=donor_node.right,
                                         age=donor_node.age, species=recipient_species,
                                         label=donor_node.label)
                active_lineages.append(introgressed_node)

        # Handle species split events
        for species, split_time in species_split_times.items():
            if split_time == current_time:
                # Update species labels for lineages
                for node in active_lineages:
                    if node.species == species:
                        node.species = f"{species}_descendant"

    # Assign mutations to the tree
    root = active_lineages[0]
    assign_mutations(root, rng, mutation_rate)

    return root

def assign_mutations(node, rng, mutation_rate):
    """
    Recursively assign mutations to the tree branches.

    Args:
        node (Node): Current node in the tree.
        rng (Generator): NumPy random number generator.
        mutation_rate (float): Mutation rate per generation.
    """
    if node.parent is not None:
        branch_length = node.age - node.parent.age
    else:
        branch_length = node.age

    expected_mutations = mutation_rate * branch_length
    num_mutations = rng.poisson(expected_mutations)
    node.mutations = [f"mut_{rng.integers(1e9)}" for _ in range(num_mutations)]

    if node.left is not None:
        assign_mutations(node.left, rng, mutation_rate)
    if node.right is not None:
        assign_mutations(node.right, rng, mutation_rate)

def collect_site_frequency_spectrum(node, total_leaves, spectrum=None):
    """
    Collect the site frequency spectrum (SFS) from the tree.

    Args:
        node (Node): Current node in the tree.
        total_leaves (int): Total number of leaf nodes.
        spectrum (dict): Dictionary to store SFS counts.

    Returns:
        set: Set of leaf labels under this node.
    """
    if spectrum is None:
        spectrum = {}

    if node.is_leaf():
        return {node.label}

    left_leaves = collect_site_frequency_spectrum(node.left, total_leaves, spectrum)
    right_leaves = collect_site_frequency_spectrum(node.right, total_leaves, spectrum)
    all_leaves = left_leaves.union(right_leaves)

    # For each mutation, count the frequency
    for mutation in node.mutations:
        freq = len(all_leaves)
        spectrum[freq] = spectrum.get(freq, 0) + 1

    return all_leaves

def print_tree(node, indent=0):
    """
    Recursively print the tree structure.

    Args:
        node (Node): Current node in the tree.
        indent (int): Current indentation level.
    """
    print(' ' * indent + f"{node}")
    if node.left is not None:
        print_tree(node.left, indent + 4)
    if node.right is not None:
        print_tree(node.right, indent + 4)

def main():
    rng = np.random.default_rng()

    # Simulation parameters
    N_e = 10000  # Effective population size
    samples_per_species = {'Species_A': 5, 'Species_B': 5}  # Number of samples per species
    species_split_times = {'Species_A': 2000.0}  # Species_A splits into descendant species at time 2000
    introgression_events = [
        {'time': 1500.0, 'donor_species': 'Species_B', 'recipient_species': 'Species_A'}
    ]  # Introgression event from Species_B to Species_A at time 1500
    mutation_rate = 1e-8  # Mutation rate per generation

    # Simulate the coalescent process with introgression
    root = simulate_coalescent_with_introgression(
        rng, N_e, samples_per_species, species_split_times, introgression_events, mutation_rate
    )

    # Print the coalescent tree
    print("Coalescent Tree:")
    print_tree(root)

    # Collect site frequency spectrum
    total_leaves = sum(samples_per_species.values())
    spectrum = {}
    collect_site_frequency_spectrum(root, total_leaves, spectrum)

    # Print site frequency spectrum
    print("\nSite Frequency Spectrum (SFS):")
    for freq in sorted(spectrum.keys()):
        print(f"Frequency {freq}/{total_leaves}: {spectrum[freq]} mutations")

if __name__ == "__main__":
    main()

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import multiprocessing as mp
from typing import Optional

class Node:
    def __init__(self, id: int, left: Optional['Node'] = None, right: Optional['Node'] = None, age: float = 0.0):
        self.id = id
        self.parent: Optional['Node'] = None
        self.left = left
        self.right = right
        self.age = age
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self

    def __repr__(self):
        return f"Node(id={self.id}, age={self.age})"

def simulate_coalescent_with_introgression(rng, N_e_func, samples_per_species, species_split_times,
                                           introgression_events, mutation_rate):
    nodes = []
    for species, num_samples in samples_per_species.items():
        nodes.extend([Node(id=f"{species}_{i}") for i in range(num_samples)])
    
    current_time = 0.0
    while len(nodes) > 1:
        rate = len(nodes) * (len(nodes) - 1) / 2
        t = rng.exponential(1 / rate)
        current_time += t
        i, j = rng.choice(len(nodes), size=2, replace=False)
        new_node = Node(id=f"ancestor_{len(nodes)}", left=nodes[i], right=nodes[j], age=current_time)
        nodes = [node for k, node in enumerate(nodes) if k not in (i, j)]
        nodes.append(new_node)
    
    return nodes[0]

def assign_mutations(node, rng, mutation_rate):
    if node.left is None and node.right is None:
        return 0
    
    branch_length = node.age - (node.parent.age if node.parent else 0)
    num_mutations = rng.poisson(branch_length * mutation_rate)
    
    if node.left:
        num_mutations += assign_mutations(node.left, rng, mutation_rate)
    if node.right:
        num_mutations += assign_mutations(node.right, rng, mutation_rate)
    
    return num_mutations

def collect_site_frequency_spectrum(node, total_leaves, spectrum=None):
    if spectrum is None:
        spectrum = defaultdict(int)
    
    if node.left is None and node.right is None:
        return 1
    
    left_count = collect_site_frequency_spectrum(node.left, total_leaves, spectrum)
    right_count = collect_site_frequency_spectrum(node.right, total_leaves, spectrum)
    
    if node.parent is None:
        spectrum[left_count] += 1
        spectrum[right_count] += 1
    else:
        spectrum[left_count] += 1
        spectrum[right_count] += 1
        spectrum[total_leaves - left_count] += 1
        spectrum[total_leaves - right_count] += 1
    
    return left_count + right_count

def run_simulation_wrapper(args):
    return run_single_simulation(*args)

def run_single_simulation(sim_id, rng_seed, N_e_func, samples_per_species, species_split_times,
                          introgression_events, mutation_rate, total_leaves):
    rng = np.random.default_rng(rng_seed)
    root = simulate_coalescent_with_introgression(rng, N_e_func, samples_per_species, species_split_times,
                                                  introgression_events, mutation_rate)
    assign_mutations(root, rng, mutation_rate)
    spectrum = collect_site_frequency_spectrum(root, total_leaves)
    newick_tree = f"({root.left.id},{root.right.id});"
    return spectrum, newick_tree

def run_simulations(num_simulations, N_e_func, samples_per_species, species_split_times,
                    introgression_events, mutation_rate):
    aggregated_spectrum = defaultdict(int)
    total_leaves = sum(samples_per_species.values())
    newick_trees = []
    args_list = []
    for sim in range(num_simulations):
        args = (
            sim,
            np.random.SeedSequence().entropy,  # Unique seed for each simulation
            N_e_func,
            samples_per_species,
            species_split_times,
            introgression_events,
            mutation_rate,
            total_leaves
        )
        args_list.append(args)
    with mp.Pool() as pool:
        results = pool.map(run_simulation_wrapper, args_list)
    for spectrum, newick_tree in results:
        newick_trees.append(newick_tree)
        for freq, count in spectrum.items():
            aggregated_spectrum[freq] += count
    return aggregated_spectrum, newick_trees

def plot_sfs_distribution(aggregated_spectrum, total_leaves, num_simulations):
    frequencies = range(1, total_leaves + 1)
    counts = [aggregated_spectrum.get(freq, 0) / num_simulations for freq in frequencies]
    plt.figure(figsize=(10, 6))
    plt.bar(frequencies, counts, color='skyblue')
    plt.xlabel('Allele Frequency')
    plt.ylabel('Average Number of Mutations')
    plt.title('Average Site Frequency Spectrum over Simulations')
    plt.xticks(frequencies)
    plt.show()

def visualize_tree(newick_tree):
    import ete3
    tree = ete3.Tree(newick_tree)
    tree.show()

def main():
    def N_e_func(time):
        if time < 1000:
            return 10000
        elif time < 2000:
            return 5000
        else:
            return 20000

    samples_per_species = {'Species_A': 5, 'Species_B': 5}
    species_split_times = {'Species_A': 2000.0}
    introgression_events = [
        {'time': 1500.0, 'donor_species': 'Species_B', 'recipient_species': 'Species_A'}
    ]
    mutation_rate = 1e-8
    num_simulations = 10

    aggregated_spectrum, newick_trees = run_simulations(
        num_simulations, N_e_func, samples_per_species,
        species_split_times, introgression_events, mutation_rate
    )

    with open('gene_trees.nwk', 'w') as f:
        for tree in newick_trees:
            f.write(tree + '\n')

    total_leaves = sum(samples_per_species.values())
    print("\nAggregated Site Frequency Spectrum (SFS) over Simulations:")
    for freq in sorted(aggregated_spectrum.keys()):
        avg_count = aggregated_spectrum[freq] / num_simulations
        print(f"Frequency {freq}/{total_leaves}: Average Mutations = {avg_count:.2f}")

    plot_sfs_distribution(aggregated_spectrum, total_leaves, num_simulations)

    for i, tree in enumerate(newick_trees):
        print(f"\nSimulation {i + 1} Coalescent Tree (Newick format):")
        print(tree)
        visualize_tree(tree)

if __name__ == "__main__":
    main()
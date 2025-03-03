class Node:

    def __init__(self, left=None, right=None, age=0.0, species=None):

        self.parent = None  # Parent node
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.age = age  # Age of the node
        self.species = species  # Species of the node

        if self.left != None:
            self.left.parent = self  # Set parent of left child

        if self.right != None:
            self.right.parent = self  # Set parent of right child

    def print_tree(self, indent=0):
        # Print tree with in-order traversal
        if self.left is not None:
            self.left.print_tree(indent + 2)

        print(' ' * indent + str(self.species))

        if self.right is not None:
            self.right.print_tree(indent + 2)


def coalesce(rng, n=10000, k=5, t_end=2000, speciation_rate=0.001, interspecies_coalescence_rate=0.0001):

    # Check input parameters for validity
    if not (n > 0 and k > 0 and t_end > 0 and 0 <= speciation_rate < 1 and 0 <= interspecies_coalescence_rate < 1):
        raise ValueError("Invalid input parameters")

    if k > n:
        raise ValueError("`k` should not be larger than `n`")

    age = 0.0  # Start time
    fourN = 4*n  # Parameter for coalescence rate
    nodes = [Node(species=i) for i in range(k)]  # Create initial nodes
    species_count = k  # Count of distinct species

    # Until end time or only one lineage left
    while k > 1 and age < t_end:

        event = rng.random()

        if event < speciation_rate:

            # Speciation event
            i = rng.integers(k)  # Pick a node
            j = rng.integers(k-1)  # Pick another node

            if j >= i:
                j += 1

            # Create a new node with a new species and the picked nodes as children
            nodes[i] = Node(nodes[i], nodes[j], age, species_count)
            species_count += 1  # Increment species count
            nodes[j] = nodes[k-1]  # Replace the second picked node
            k -= 1  # Decrement number of lineages

        elif event < speciation_rate + interspecies_coalescence_rate:

            # Coalescence event between different species
            i, j = rng.choice(range(k), 2, replace=False)  # Pick two nodes

            # Coalesce the nodes into a new node
            nodes[i] = Node(nodes[i], nodes[j], age)
            nodes[j] = nodes[k-1]  # Replace the second picked node

            k -= 1  # Decrement number of lineages

        else:
            # Coalescence event within same species
            same_species_nodes = [idx for idx, node in enumerate(
                nodes[:k]) if node.species == nodes[0].species]

            if len(same_species_nodes) < 2:
                continue  # If no pair of nodes from same species, skip to next loop

            # Pick two nodes from same species
            i, j = rng.choice(same_species_nodes, 2, replace=False)
            # Coalesce the nodes into a new node
            nodes[i] = Node(nodes[i], nodes[j], age)
            nodes[j] = nodes[k-1]  # Replace the second picked node
            k -= 1  # Decrement number of lineages

        age += rng.exponential(fourN / (k*(k-1)))  # Increment age

    while k > 1:

        # Continue coalescence events until only one lineage is left
        age, k = coalescent_step(rng, age, fourN, k, nodes)

    return nodes[0]  # Return the root of the coalescent tree


def coalescent_step(rng, age, fourN, k, nodes):

    tmean = fourN / (k*(k-1))  # Mean waiting time until next coalescent event
    age += rng.exponential(tmean)  # Increment age
    i = rng.integers(k)  # Pick a node
    j = rng.integers(k-1)  # Pick another node

    if j >= i:
        j += 1

    # Coalesce the nodes into a new node
    nodes[i] = Node(nodes[i], nodes[j], age)
    nodes[j] = nodes[k-1]  # Replace the second picked node
    k -= 1  # Decrement number of lineages

    return age, k  # Return updated age and number of lineages


def ndescendants(node):

    # Count the number of descendants of a node
    stack = [node]  # Stack for depth-first traversal
    count = 0  # Count of descendants

    while stack:

        node = stack.pop()  # Get next node from stack
        count += 1  # Increment count

        if node.left:
            stack.append(node.left)  # Add left child to stack

        if node.right:
            stack.append(node.right)  # Add right child to stack

    return count


def spectrum(node, rng, mu, spec):

    # Count the number of mutations along each lineage
    # Stack for depth-first traversal
    stack = [(node, node.parent.age if node.parent else 0.0)]

    while stack:

        node, parent_age = stack.pop()  # Get next node and its parent's age from stack
        brlen = parent_age - node.age  # Length of branch to parent
        nmut = rng.poisson(mu * brlen)  # Number of mutations along branch
        nkids = 1  # Number of kids

        if node.left:
            stack.append((node.left, node.age))  # Add left child to stack
            nkids += 1

        if node.right:
            stack.append((node.right, node.age))  # Add right child to stack
            nkids += 1

        spec[nkids-1] += nmut  # Increment mutation count

    return nkids


def main():
    import numpy as np
    rng = np.random.default_rng()  # Create random number generator
    root = coalesce(rng)  # Run coalescent simulation
    root.print_tree()
    spec = [0]*(ndescendants(root)-1)  # Initialize mutation spectrum
    spectrum(root, rng, 1e-8, spec)  # Count mutations along each lineage
    print("SFS:", spec)  # Print site frequency spectrum


if __name__ == "__main__":
    main()

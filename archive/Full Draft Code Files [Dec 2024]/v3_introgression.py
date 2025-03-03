
class Node:

    # default age will be 0
    def __init__(self, left=None, right=None, age=0.0):
        self.parent = None  # Parent node
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.age = age  # Age of the node
        if self.left != None:
            self.left.parent = self  # Set parent of left child
        if self.right != None:
            self.right.parent = self  # Set parent of right child


# random number, n = diploid pop sie, k = haploid sample size (number of gene copies)
# Simulate Introgression by transferring information from one population into this population
# only simulates it as a one time event into the main population.
# runs the coalescent process independently for both populations - until the specified time - 't_introgression'
# at this time, it will select a pair of nodes from each population - then it coalesces them into a single node.

# main pop has n individuals, hence 2n haploid gene copies - then you sample k haploid gene copies from it.
# Introgressed pop has ni individuals, hence 2ni gene copies - sample ki haploid gene copies from it.

# backcross_rate: prob of backcrossing at each time step.

def coalesce_with_introgression(rng, n=10000, k=5, ni=2000, ki=2, t_introgression=1000, t_end=2000, backcross_rate=0.01):
    age = 0.0  # time in generations
    fourN = 4*n
    fourNi = 4*ni

    # create a list of nodes for the main population
    nodes = [Node() for i in range(k)]
    nodes_introgression = [Node() for i in range(ki)]

    # perform the coalescent process separately, until it reaches the time of the introgression event.
    while k > 1 and age < t_introgression:
        age, k = coalescent_step(rng, age, fourN, k, nodes)

    while ki > 1 and age < t_introgression:
        age, ki = coalescent_step(rng, age, fourNi, k, nodes_introgression)

    # backcrossing events start from t_introgression until t_end
    while age < t_end:

        # backcrossing only happens when theres > 1 lineage in the main pop and at least 1 in the introgressed pop

        if rng.random() < backcross_rate and k > 1 and ki > 0:

            # select a random pair of nodes from the main and the introgression populations
            i = rng.integers(k)
            j = rng.integers(ki)

            # coalesces a node from main w node from introgressed pop.
            nodes[i] = Node(nodes[i], nodes_introgression[j], age)
            if ki > 1:

                # replacing the introgressed node with the last node.
                nodes_introgression[j] = nodes_introgression[ki-1]
            ki -= 1

        # if backcrossing does not occur, then coalescence step occurs in the main pop.
        else:
            age, k = coalescent_step(rng, age, fourN, k, nodes)

    # continue the coalescent process in the main pop until you reach the root node.
    while k > 1:
        age, k = coalescent_step(rng, age, fourN, k, nodes)

    return nodes[0]

# rng = random number generator.
# age is the current age
# k is the current number of lineages
# haploid population means that every organism has one SET of chromosomes = 2
# so the rate of coalescence is 1/2N : because 1/2N that means 2 have the same chromosome
# for diploid organisms - means organisms w 2 sets of chromosomes, so rate of coalescence is doubled.


def coalescent_step(rng, age, fourN, k, nodes):

    # average time until the next coalescent event is computed
    # depends on k (no. of lineages present), and N (pop size)
    tmean = fourN / (k*(k-1))

    # time until the next coalescent event is computed
    age += rng.exponential(tmean)

    # choose the indices of the lineages in the list 'nodes'
    i = rng.integers(k)
    j = rng.integers(k-1)

    # make sure that lineages are not the same
    # If they are, set j to the last index.
    if j == i:
        j = k-1

    # make sure that i is always less than j
    if j < i:
        i, j = j, i

    # replace the ith lineage with the new coalesced node.
    nodes[i] = Node(nodes[i], nodes[j], age)

    # if jth lineage is not the last lineage
    if j < k-1:

        # move the last lineage to the position of the jth lineage.
        # this discards the jth lineage - because it already coalesced.
        nodes[j] = nodes[k-1]

    # number of lineages decreases by 1
    k -= 1
    return age, k

# ndescendents and spectrum remain the same as v2 - this creates the structure for the genealogy.

# calc the no. of tip nodes that are descendants of a given node


def ndescendants(node):

    # this is only true for tip nodes
    # a tip node will only have 1 descendant - itself
    if node.left == None:
        return 1
    else:
        # else you will be at a root or internal node
        # so the number of descendants is the sum of descendants on the left and right children.
        # when you call it recursively - you will sum all the descendants of all the nodes

        return ndescendants(node.left) + ndescendants(node.right)

# A site frequency spectrum (SFS) describes the distribution of allele frequencies
# across sites in the genome of a particular species.
# the behaviour of spectrum is dependent on whether the node is a root / tip /internal
# basically calculates the expected number of mutations for each node.


def spectrum(node, rng, mu, spec):

    # NODE in the initial call - node will refer to the root of the gene genealogy
    # the function will call itself recursively to traverse the entire gene genealogy.
    # RNG will generate mutations by sampling from the Poisson distribution
    # MU is the mutation rate: a floating pt number
    # SPEC: an array of integers: the length will be k-1
    # k is the no. o gene copies in the sample.
    # initial call: all entries of spec will be == 0.
    # spec[i] will == the number of mutations that have i+1 mutations within the sample.

    # ROOT NODE
    if node.parent == None:
        return spectrum(node.left, rng, mu, spec) + spectrum(node.right, rng, mu, spec)

    # length of the branch connecting the current node to the parent.
    brlen = node.parent.age - node.age

    # expected no. of mutations = mu * brlen
    nmut = rng.poisson(mu * brlen)

    # TIP NODE
    if node.left == None:
        nkids = 1  # refers to only itself.

    # INTERNAL NODE
    else:
        # count no. of descendants for each child
        nkids = spectrum(node.left, rng, mu, spec) + \
            spectrum(node.right, rng, mu, spec)

    # mutation count is added to the spec array.
    spec[nkids-1] += nmut

    return nkids

# Define main function to simulate and print


def main():
    import numpy as np
    rng = np.random.default_rng()
    root = coalesce_with_introgression(rng)
    spec = [0]*(ndescendants(root)-1)
    spectrum(root, rng, 1e-8, spec)
    print("SFS:", spec)


if __name__ == "__main__":
    main()

# create Node
class Node:

    # default age will be 0
    def __init__(self, left=None, right=None, age=0.0):
        self.parent = None
        self.left = left
        self.right = right
        self.age = age
        if self.left != None:
            self.left.parent = self
        if self.right != None:
            self.right.parent = self

# random number, n = diploid pop sie, k = haploid sample size (number of gene copies)


def coalesce(rng, n=10000, k=5):
    age = 0.0  # time in generations
    fourN = 4*n

    # create a list of nodes
    nodes = [Node() for i in range(k)]

    # each iteration through the loop is one coalescent interval
    while k > 1:
        tmean = fourN/(k*(k-1))

        # time until the next coalescent event
        # draw a random value from the exponential distribution
        age += rng.exponential(tmean)

        # choose i and j, ensure that i < j
        i = rng.integers(k)
        j = rng.integers(k-1)
        if j == i:
            j = k-1
        if j < i:
            i, j = j, i

        # ASK WHAT IS GOING ON HERE
        # join  i and j - put the parent into position i within the array of nodes
        nodes[i] = Node(nodes[i], nodes[j], age)

        # shorten the array and forget about node j
        if j < k-1:
            nodes[j] = nodes[k-1]

        k -= 1

    # while loop ends when k = 1
    # root node is returned
    return nodes[0]

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

    # if current node is an internal node is a tip then it has no children.
    # is it is an internal node - you need to call spectrum on the left and right children to det nkids.

    spec[nkids-1] += nmut

    # length of the branch connecting the current node to the parent.
    brlen = node.parent.age - node.age

    # expected no. of mutations = mu * brlen
    nmut = rng.poisson(mu * brlen)

    # ROOT NODE
    if node.parent == None:
        return spectrum(node.left) + spectrum(node.right)

    else:

        # TIP NODE
        if node.left == None:
            nkids = 1  # refers to only itself.

        # INTERNAL NODE
        else:
            # count no. of descendants for each child
            nkids = spectrum(node.left) + spectrum(node.right)

    # mutation count is added to the spec array.
    spec[nkids-1] += nmut

    return nkids

import random
from anytree import Node as AnyNode, RenderTree, AsciiStyle

class Node:
    def __init__(self, name, left=None, right=None, age=0.0):
        self.name = name
        self.parent = None
        self.left = left
        self.right = right
        self.age = age
        if self.left != None:
            self.left.parent = self
        if self.right != None:
            self.right.parent = self

def coalesce_samples(n=10, random_seed=42):
    random.seed(random_seed)
    fourN = 4 * n
    nodes = [Node(name=str(i), age=0) for i in range(1, n+1)]
    age = 0
    while len(nodes) > 1:
        tmean = fourN / (len(nodes) * (len(nodes) - 1))
        age += random.expovariate(1 / tmean)
        i, j = random.sample(range(len(nodes)), 2)
        new_node = Node(name=str(max(int(node.name) for node in nodes)+1), left=nodes[i], right=nodes[j], age=age)
        nodes = [node for idx, node in enumerate(nodes) if idx not in {i, j}] + [new_node]
    return nodes[0]

def to_anytree(node):
    any_node = AnyNode(name=node.name, age=node.age)
    children = []
    if node.left is not None:
        children.append(to_anytree(node.left))
    if node.right is not None:
        children.append(to_anytree(node.right))
    any_node.children = tuple(children)
    return any_node

root = coalesce_samples()
any_root = to_anytree(root)

for pre, fill, node in RenderTree(any_root):
    print("%s%s" % (pre, node.name))

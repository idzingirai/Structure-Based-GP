import random

from genetic_program.config import MAX_TREE_DEPTH
from src.tree.node import Node
from src.tree.tree_util import *


def prune_tree(root: Node, max_depth: int, tree_generator: BinaryTreeGenerator) -> Optional[Node]:
    if root is None:
        return None

    if max_depth == 0:
        return tree_generator.full(1)

    root.left = prune_tree(root.left, max_depth - 1, tree_generator)
    root.right = prune_tree(root.right, max_depth - 1, tree_generator)

    return root

def crossover(first_tree: Node, second_tree: Node, tree_generator: BinaryTreeGenerator) -> None:
    num_of_nodes_in_first_tree = get_number_of_nodes(first_tree)
    num_of_nodes_in_second_tree = get_number_of_nodes(second_tree)

    first_subtree_index = random.randint(1, (num_of_nodes_in_first_tree - 1) // 2)
    second_subtree_index = random.randint(1, (num_of_nodes_in_second_tree - 1) // 2)

    first_subtree = get_node_by_index(first_tree, first_subtree_index)
    second_subtree = get_node_by_index(second_tree, second_subtree_index)

    first_subtree_parent_index = (first_subtree_index - 1) // 2
    second_subtree_parent_index = (second_subtree_index - 1) // 2

    first_parent = get_node_by_index(first_tree, first_subtree_parent_index)
    if first_subtree_index % 2 != 0:
        first_parent.left = second_subtree
    else:
        first_parent.right = second_subtree

    second_parent = get_node_by_index(second_tree, second_subtree_parent_index)
    if second_subtree_index % 2 != 0:
        second_parent.left = first_subtree
    else:
        second_parent.right = first_subtree

    first_offspring_depth = get_tree_depth(first_tree)
    second_offspring_depth = get_tree_depth(second_tree)

    if first_offspring_depth > MAX_TREE_DEPTH:
        prune_tree(first_tree, MAX_TREE_DEPTH, tree_generator)

    if second_offspring_depth > MAX_TREE_DEPTH:
        prune_tree(second_tree, MAX_TREE_DEPTH, tree_generator)


def mutation(tree: Node, tree_generator: BinaryTreeGenerator) -> None:
    num_of_nodes_in_tree = get_number_of_nodes(tree)
    mutation_point = random.randint(1, num_of_nodes_in_tree - 1)

    node = get_node_by_index(tree, mutation_point)
    node_level = get_node_level_by_index(node, mutation_point)
    subtree_depth = random.randint(1, MAX_TREE_DEPTH - node_level)

    if random.randint(0, 1) == 0:
        subtree = tree_generator.full(subtree_depth)
    else:
        subtree = tree_generator.grow(subtree_depth)

    mutation_point_parent_index = (mutation_point - 1) // 2
    mutation_point_parent = get_node_by_index(tree, mutation_point_parent_index)

    if mutation_point % 2 != 0:
        mutation_point_parent.left = subtree
    else:
        mutation_point_parent.right = subtree

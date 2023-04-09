from genetic_program.config import INITIAL_TREE_DEPTH, POPULATION_SIZE, TREE_GENERATION_METHOD

from tree.binary_tree_generator import BinaryTreeGenerator
from tree.node import Node


def generate_initial_population(tree_generator: BinaryTreeGenerator) -> [Node]:
    initial_tree_depth = INITIAL_TREE_DEPTH
    population_size = POPULATION_SIZE
    tree_generation_method = TREE_GENERATION_METHOD

    population = []
    num_of_trees_at_each_level = int(population_size / (initial_tree_depth - 1))

    for depth in range(2, initial_tree_depth + 1):
        for i in range(num_of_trees_at_each_level):
            if tree_generation_method == 'RAMPED':
                population.append(tree_generator.ramped(depth, i))
            elif tree_generation_method == 'FULL':
                population.append(tree_generator.full(depth))
            else:
                population.append(tree_generator.grow(depth))

    return population

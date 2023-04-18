import math
from typing import List

from src.genetic_program.config import *
from src.genetic_program.fitness import calculate_fitness
from src.tree.tree_generator import BinaryTreeGenerator
from src.tree.tree_util import *


class GeneticProgram:
    def __init__(self, terminal_set: list):
        #   global parameters
        self._tree_generator = BinaryTreeGenerator(terminal_set=terminal_set)
        self._terminal_set = terminal_set

        #   For generating the population
        self._population_size = POPULATION_SIZE
        self._initial_tree_depth = INITIAL_TREE_DEPTH
        self._tree_generation_method = TREE_GENERATION_METHOD

        #   For selection
        self._tournament_size = TOURNAMENT_SIZE

        #   For crossover and mutation
        self._max_tree_depth = MAX_TREE_DEPTH
        self._crossover_probability = CROSSOVER_RATE
        self._mutation_probability = MUTATION_RATE

        #   Data for the fitness function
        self._x_data = None
        self._y_data = None

    def set_data(self, x_data, y_data):
        self._x_data = x_data
        self._y_data = y_data

    def generate_population(self) -> List[Node]:
        """
        Generates a population of trees based on the parameters set in the config file.
        :return: a list of trees
        """
        population = []
        num_of_trees_at_each_level = math.floor(self._population_size / (self._initial_tree_depth - 1))

        for depth in range(2, self._initial_tree_depth + 1):
            for index in range(num_of_trees_at_each_level):
                if self._tree_generation_method == 'RAMPED':
                    population.append(self._tree_generator.ramped(depth=depth, index_on_level=index))
                elif self._tree_generation_method == 'FULL':
                    population.append(self._tree_generator.full(depth=depth))
                else:
                    population.append(self._tree_generator.grow(depth=depth))

        return population

    def generate_population_with_fixed_component(self, component: Node) -> List[Node]:
        """
        Generates a population of trees based on the parameters set in the config file.
        :return: a list of trees
        """
        pass

    def tournament_selection(self, population: list) -> Node:
        """
        Selects a tree from the population using tournament selection.
        :param population:
        :return: a randomly selected tree from the population
        """
        tournament = random.sample(population, self._tournament_size)
        tournament.sort(key=lambda t: t.fitness)
        return clone(root=tournament[0])

    def crossover(self, first_tree: Node, second_tree: Node) -> None:
        """
        Performs crossover on two trees.
        :param first_tree:
        :param second_tree:
        :return: nothing
        """
        num_of_nodes_in_first_tree = count_nodes(root=first_tree)
        num_of_nodes_in_second_tree = count_nodes(root=second_tree)

        first_subtree_index = random.randint(1, math.floor((num_of_nodes_in_first_tree - 1) / 2))
        second_subtree_index = random.randint(1, math.floor((num_of_nodes_in_second_tree - 1) / 2))

        first_subtree = get_node(root=first_tree, index=first_subtree_index)
        second_subtree = get_node(root=second_tree, index=second_subtree_index)

        first_subtree_parent_index = math.floor((first_subtree_index - 1) / 2)
        second_subtree_parent_index = math.floor((second_subtree_index - 1) / 2)

        first_parent = get_node(root=first_tree, index=first_subtree_parent_index)
        if first_subtree_index % 2 != 0:
            first_parent.left = second_subtree
        else:
            first_parent.right = second_subtree

        second_parent = get_node(root=second_tree, index=second_subtree_parent_index)
        if second_subtree_index % 2 != 0:
            second_parent.left = first_subtree
        else:
            second_parent.right = first_subtree

        first_offspring_depth = get_depth(root=first_tree)
        second_offspring_depth = get_depth(root=second_tree)

        if first_offspring_depth > self._max_tree_depth:
            prune(root=first_tree, max_depth=self._max_tree_depth, terminal_set=self._terminal_set)

        if second_offspring_depth > self._max_tree_depth:
            prune(root=second_tree, max_depth=self._max_tree_depth, terminal_set=self._terminal_set)

    def mutation(self, tree: Node) -> None:
        """
        Mutates a tree by replacing a random subtree with a new subtree.
        :param tree:
        :return: nothing
        """
        num_of_nodes_in_tree = count_nodes(root=tree)
        mutation_point = random.randint(1, num_of_nodes_in_tree - 1)

        node = get_node(root=tree, index=mutation_point)
        node_level = get_node_level(root=node, index=mutation_point)
        subtree_depth = random.randint(1, self._max_tree_depth - node_level)

        if random.randint(0, 1) == 0:
            subtree = self._tree_generator.full(depth=subtree_depth)
        else:
            subtree = self._tree_generator.grow(depth=subtree_depth)

        mutation_point_parent_index = (mutation_point - 1) // 2
        mutation_point_parent = get_node(root=tree, index=mutation_point_parent_index)

        if mutation_point % 2 != 0:
            mutation_point_parent.left = subtree
        else:
            mutation_point_parent.right = subtree

    def run(self):
        population = self.generate_population()

        for tree in population:
            calculate_fitness(tree=tree, x=self._x_data, y=self._y_data)

        population.sort(key=lambda t: t.fitness)

        first_tree = self.tournament_selection(population=population)
        second_tree = self.tournament_selection(population=population)

        if random.random() < self._crossover_probability:
            self.crossover(first_tree=first_tree, second_tree=second_tree)

        if random.random() < self._mutation_probability:
            self.mutation(tree=first_tree)
            self.mutation(tree=second_tree)

        calculate_fitness(tree=first_tree, x=self._x_data, y=self._y_data)
        calculate_fitness(tree=second_tree, x=self._x_data, y=self._y_data)

        best_tree = first_tree if first_tree.fitness > second_tree.fitness else second_tree
        return clone(best_tree) if best_tree.fitness > population[0].fitness else clone(population[0])

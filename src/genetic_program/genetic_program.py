import math

from src.genetic_program.config import *
from src.genetic_program.fitness import calculate_fitness
from src.genetic_program.isba import calculate_gsim, calculate_lsim
from src.tree.tree_generator import BinaryTreeGenerator
from src.tree.tree_util import *


class GeneticProgram:
    def __init__(self, terminal_set: list):
        #   global parameters
        self._tree_generator = BinaryTreeGenerator(terminal_set=terminal_set)
        self._terminal_set = terminal_set

        #  For the algorithm
        self._max_num_of_global_optima = 20
        self._max_num_of_generations = MAX_GENERATIONS

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
        self._x = None
        self._y = None

    def set_data(self, x, y):
        self._x = x
        self._y = y

    def generate_individual(self, depth: int) -> Node:
        """
        Generates a tree with depth d.
        :param depth:
        :return: root node of the tree
        """
        if self._tree_generation_method == 'RAMPED':
            return self._tree_generator.full(depth=depth)
        elif self._tree_generation_method == 'FULL':
            return self._tree_generator.full(depth=depth)
        else:
            return self._tree_generator.grow(depth=depth)

    def generate_individual_with_fixed_component(self, component, depth) -> Node:
        """
        Generates an individual with a fixed component.
        :param component:
        :param depth:
        :return: root node of the tree
        """
        tree = clone(component)
        mutation_point = get_first_leaf_index(tree)
        individual = self.generate_individual(depth=depth)

        mutation_point_parent_index = (mutation_point - 1) // 2
        mutation_point_parent = get_node(root=tree, index=mutation_point_parent_index)

        if mutation_point % 2 != 0:
            mutation_point_parent.left = individual
        else:
            mutation_point_parent.right = individual

        return tree

    def generate_population(self) -> List[Node]:
        """
        Generates a population of trees.
        :return: population of trees
        """
        population = []
        num_of_trees_at_each_level = math.floor(self._population_size / (self._initial_tree_depth - 1))

        for depth in range(2, self._initial_tree_depth + 1):
            for index in range(num_of_trees_at_each_level):
                population.append(self.generate_individual(depth=depth))

        return population

    def generate_population_with_fixed_component(self, component: Node) -> List[Node]:
        """
        Generates a list of trees with a fixed component.
        :param component:
        :return: list of trees with a fixed component
        """
        population = []
        num_of_trees_at_each_level = math.floor(self._population_size / (self._initial_tree_depth - 1))

        for depth in range(2, self._initial_tree_depth + 1):
            for index in range(num_of_trees_at_each_level):
                population.append(self.generate_individual_with_fixed_component(component, depth))

        return population

    def tournament_selection(self, population: list) -> Node:
        """
        Selects a tree from a sample of the population.
        :param population:
        :return: a randomly selected tree from the population
        """
        tournament = random.sample(population, self._tournament_size)
        tournament.sort(key=lambda t: t.fitness)
        return clone(root=tournament[0])

    def crossover(self, first_tree: Node, second_tree: Node) -> None:
        """
        Performs subtree crossover on two trees.
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

    def global_run(self, local_optima: List[Node]) -> Node:
        """
        Performs a genetic programming run and returns the best tree found.
        :return: the best tree found
        """
        # Generate initial population
        population = self.generate_population()

        # Calculate fitness of each individual and remove those that are similar to local optima
        for individual in population:
            calculate_fitness(individual, self._x, self._y)

            if calculate_gsim(individual, local_optima, 0.5, 0.5, 2) and individual.fitness < 3:
                population.remove(individual)

        # Sort population by fitness
        population.sort(key=lambda t: t.fitness)

        # Save best tree
        best_tree = clone(population[0])

        # Run algorithm
        num_of_generations = 0
        while num_of_generations < self._max_num_of_generations:
            #  Generate offspring by applying selection, crossover and mutation
            first_tree, second_tree = self.generate_offspring(population)

            # Calculate fitness of offspring
            calculate_fitness(first_tree, self._x, self._y)
            calculate_fitness(second_tree, self._x, self._y)

            #  Added offspring to population if they are not similar to local optima
            if not calculate_gsim(first_tree, local_optima, 0.5, 0.5, 2):
                population.append(first_tree)

            if not calculate_gsim(second_tree, local_optima, 0.5, 0.5, 2):
                population.append(second_tree)

            # Sort population by fitness
            population.sort(key=lambda t: t.fitness)

            # Remove the worst individuals
            while len(population) > self._population_size:
                population.pop()

            # Update best tree
            if population[0].fitness < best_tree.fitness:
                best_tree = clone(population[0])

            num_of_generations += 1

        return best_tree

    def generate_offspring(self, population):
        """
        Generates two offspring from the population.
        :param population:
        :return:
        """
        # Selection
        first_tree = self.tournament_selection(population)
        second_tree = self.tournament_selection(population)

        # Crossover
        if random.random() < self._crossover_probability:
            self.crossover(first_tree, second_tree)

        # Mutation
        if random.random() < self._mutation_probability:
            self.mutation(first_tree)
            self.mutation(second_tree)

        return first_tree, second_tree

    def is_similar(self, tree: Node, local_optima: List[Node]) -> bool:
        """
        Checks if a tree is similar to any of the local optima.
        :param tree:
        :param local_optima:
        :return: True if the tree is similar to any of the local optima, False otherwise
        """
        for local_optimum in local_optima:
            if calculate_lsim(tree, local_optimum, 3):
                return True

        return False

    def local_run(self, local_optima: List[Node], fixed_component: Node) -> Optional[Node]:
        """
        Performs a genetic programming run and returns the best tree found.
        :param local_optima:
        :param fixed_component:
        :return: node
        """

        # Generate initial population
        population = self.generate_population_with_fixed_component(fixed_component)

        # Calculate fitness of each individual and remove those that are similar to local optima
        for individual in population:
            calculate_fitness(individual, self._x, self._y)

            if self.is_similar(individual, local_optima) and individual.fitness > 3:
                population.remove(individual)

        if len(population) == 0:
            return None

        if len(population) < self._tournament_size:
            return population[0]

        # Sort population by fitness
        population.sort(key=lambda t: t.fitness)

        # Save best tree
        best_tree = clone(population[0])

        # Run algorithm
        num_of_generations = 0
        while num_of_generations < self._max_num_of_generations:
            #  Generate offspring by applying selection, crossover and mutation
            first_tree, second_tree = self.generate_offspring(population)

            # Calculate fitness of offspring
            calculate_fitness(first_tree, self._x, self._y)
            calculate_fitness(second_tree, self._x, self._y)

            #  Added offspring to population if they are not similar to local optima
            is_first_tree_similar = False
            is_second_tree_similar = False

            for local_optimum in local_optima:
                if calculate_lsim(first_tree, local_optimum, 3):
                    is_first_tree_similar = True
                    break

            for local_optimum in local_optima:
                if calculate_lsim(second_tree, local_optimum, 3):
                    is_second_tree_similar = True
                    break

            if not is_first_tree_similar:
                population.append(first_tree)

            if not is_second_tree_similar:
                population.append(second_tree)

            # Sort population by fitness
            population.sort(key=lambda t: t.fitness)

            # Remove the worst individuals
            while len(population) > self._population_size:
                population.pop()

            # Update best tree
            if population[0].fitness < best_tree.fitness:
                best_tree = clone(population[0])

            num_of_generations += 1

        return best_tree

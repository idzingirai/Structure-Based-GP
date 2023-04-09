import os
import time

from data.get_data import DataGetter
from genetic_program.population_generation import generate_initial_population
from genetic_program.fitness import calculate_fitness
from genetic_program.selection import tournament_selection
from tree.tree_util import copy_tree, get_tree_postfix_expr
from tree.binary_tree_generator import BinaryTreeGenerator

unprocessed_file_path = os.path.join('../data/interim/For_modeling.csv')
processed_file_path = os.path.join('../data/processed/For_modeling.parquet')

start_time = time.time()

#   Get the data
data_getter = DataGetter(unprocessed_file_path, processed_file_path)
x_train, x_test, y_train, y_test = data_getter.get_data()
selected_features = data_getter.get_selected_features()

print(f"[] Features: {selected_features}\n")

tree_generator = BinaryTreeGenerator(length_of_terminal_set=len(selected_features))
population = generate_initial_population(tree_generator)

for tree in population:
    calculate_fitness(tree, x_train, y_train)

population.sort(key=lambda t: t.fitness)
best_tree = copy_tree(population[0])
calculate_fitness(best_tree, x_train, y_train)

num_of_generations = 0
num_of_generations_without_improvement = 0

while num_of_generations < 10:
    first_tree = tournament_selection(population)
    second_tree = tournament_selection(population)

    # if random.random() < CROSSOVER_RATE:
    #     crossover(first_tree, second_tree, tree_generator)
    #
    # if random.random() < MUTATION_RATE:
    #     mutation(first_tree, tree_generator)
    #     mutation(second_tree, tree_generator)
    #
    # first_offspring = copy_tree(first_tree)
    # second_offspring = copy_tree(second_tree)
    #
    # calculate_fitness(first_offspring, x_train, y_train)
    # calculate_fitness(second_offspring, x_train, y_train)
    #
    # population.append(first_offspring)
    # population.append(second_offspring)
    #
    # population.sort(key=lambda t: t.fitness)
    # population.pop()
    # population.pop()
    #
    # if population[0].fitness < best_tree.fitness:
    #     best_tree = copy_tree(population[0])
    #     calculate_fitness(best_tree, x_train, y_train)
    #     num_of_generations_without_improvement = 0
    # else:
    #     num_of_generations_without_improvement += 1
    #
    # if num_of_generations_without_improvement == 5:
    #     new_population = generate_initial_population(tree_generator)
    #     for tree in new_population:
    #         calculate_fitness(tree, x_train, y_train)
    #
    #     population.extend(new_population)
    #     population.sort(key=lambda t: t.fitness)
    #
    #     del population[100:]
    num_of_generations += 1

print(f"[] Best Program After Training Fitness {'FITNESS_FUNCTION'}: " + str(round(best_tree.fitness, 4)) + "\n")
calculate_fitness(best_tree, x_test, y_test)
print(f"[] Best Program After Testing Fitness {'FITNESS_FUNCTION'}: " + str(round(best_tree.fitness, 4)) + "\n")
print(f"[] Equation In Postfix Notation: ", get_tree_postfix_expr(best_tree))

print("\n[] Time Elapsed: " + str(time.time() - start_time) + " seconds")








import os
import random
import string
import time

from data.get_data import DataGetter
from genetic_program.genetic_program import GeneticProgram
from src.tree.tree_util import clone, prune

if __name__ == '__main__':
    #   Seed the program
    seed = 42
    random.seed(seed)

    unprocessed_file_path = os.path.join('../data/interim/For_modeling.csv')
    processed_file_path = os.path.join('../data/processed/For_modeling.parquet')

    start_time = time.time()

    #   Get the data
    data_getter = DataGetter(unprocessed_file_path, processed_file_path)
    x_train, x_test, y_train, y_test = data_getter.get_data()
    selected_features = data_getter.get_selected_features()
    placeholder_variables = list(string.ascii_lowercase)[:len(selected_features)]

    # Instantiate the genetic program
    algorithm = GeneticProgram(placeholder_variables)
    algorithm.set_data(x_train, y_train)

    #   Run the algorithm
    fitness_threshold = 3

    best_solution = None

    global_optima = []
    num_of_global_areas_explored = 0
    m = 10

    best_local_optimum = None
    best_global_optimum = None

    while num_of_global_areas_explored < m:
        # Perform a run
        global_optimum = algorithm.global_run(global_optima)

        # Check if the global optimum is good enough
        if global_optimum.fitness < fitness_threshold:
            best_solution = global_optimum
            break

        #   Add the global optimum to the list of global optima
        global_optimum_clone = clone(global_optimum)
        global_optima.append(global_optimum_clone)

        #   Get the fixed component
        fixed_component = prune(global_optimum_clone, 4, placeholder_variables)

        #   Set the parameters for the local search
        local_optima = []
        num_of_local_areas_explored = 0
        n = 5

        while num_of_local_areas_explored < n:
            #   Perform a run
            local_optimum = algorithm.local_run(local_optima, fixed_component)

            #   Check if local optimum is exists
            if local_optimum is None:
                continue

            #   Check if the local optimum is good enough
            if local_optimum.fitness < fitness_threshold:
                best_solution = local_optimum
                break

            #   Add the local optimum to the list of local optima
            local_optimum_clone = clone(local_optimum)
            local_optima.append(local_optimum_clone)

            if best_local_optimum is None or local_optimum.fitness < best_local_optimum.fitness:
                best_local_optimum = local_optimum

            #   Increment the number of local areas explored
            num_of_local_areas_explored += 1
            print("[] Local areas explored: ", num_of_local_areas_explored)

        #   Increment the number of global areas explored
        num_of_global_areas_explored += 1
        print("[] Global areas explored: ", num_of_global_areas_explored)

        if best_global_optimum is None or global_optimum.fitness < best_global_optimum.fitness:
            best_global_optimum = global_optimum

    print("[] Best local fitness: ", best_local_optimum.fitness)
    print("[] Best global fitness: ", best_global_optimum.fitness)
    print(f"[] Time taken to get the data: {time.time() - start_time} seconds\n")

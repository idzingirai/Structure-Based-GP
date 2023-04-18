import os
import random
import string
import time

from data.get_data import DataGetter
from genetic_program.genetic_program import GeneticProgram

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

# Instantiate the algorithm
algorithm = GeneticProgram(placeholder_variables)
algorithm.set_data(x_train, y_train)

#   Run the algorithm
best_solution = None
global_areas_visited = 0

print(f"[] Time taken to get the data: {time.time() - start_time} seconds\n")

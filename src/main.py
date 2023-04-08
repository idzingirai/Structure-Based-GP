import os
import time

from data.get_data import DataGetter

unprocessed_file_path = os.path.join('../data/interim/For_modeling.csv')
processed_file_path = os.path.join('../data/processed/For_modeling.parquet')

start_time = time.time()

data_getter = DataGetter(unprocessed_file_path, processed_file_path)
x_train, x_test, y_train, y_test = data_getter.get_data()
selected_features = data_getter.get_selected_features()

print(f"[] Features: {selected_features}\n")


print("\n[] Time Elapsed: " + str(time.time() - start_time) + " seconds")








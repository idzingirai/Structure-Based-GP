import os.path
import cupy as cp
import pandas as pd

from sklearn.model_selection import train_test_split

import data.config as config

from data.process import DataProcessor


class DataGetter:
    def __init__(self, unprocessed_file_path, processed_file_path):
        if not os.path.exists(processed_file_path):
            processor = DataProcessor(unprocessed_file_path)
            processor.process()
            processor.save_data(processed_file_path)

        self.df = pd.read_parquet(processed_file_path)
        self.columns = None

    def get_selected_features(self):
        return list(self.columns)

    def get_data(self):
        x = self.df.drop('Duration', axis=1)
        y = self.df['Duration']

        self.columns = x.columns

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        return cp.asarray(x_train), cp.asarray(x_test), cp.asarray(y_train), cp.asarray(y_test)

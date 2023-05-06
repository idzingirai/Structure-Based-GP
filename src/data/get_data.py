import os.path

import pandas as pd
from data.config import RANDOM_STATE, TEST_SIZE
from data.process import DataProcessor
from sklearn.model_selection import train_test_split


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

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

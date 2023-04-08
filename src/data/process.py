from data.config import THRESHOLD, SAMPLE_SIZE, RANDOM_STATE, NUM_OF_FEATURES

import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression


class DataProcessor:
    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path, engine='pyarrow')

    def process(self) -> None:
        #   Remove outliers
        z_scores = stats.zscore(self.df)
        outliers = (abs(z_scores) > THRESHOLD).any(axis=1)
        self.df = self.df[~outliers]

        #   Perform Feature Selection
        x = self.df.drop('Duration', axis=1)
        y = self.df['Duration']

        selector = SelectKBest(f_regression, k=NUM_OF_FEATURES)
        selector.fit_transform(x, y)
        selected_features = x.columns[selector.get_support()]
        selected_features = list(selected_features)
        selected_features.append('Duration')

        #   Remove unwanted features
        columns_to_drop = [col for col in self.df.columns if col not in selected_features]
        self.df = self.df.drop(columns_to_drop, axis=1)

    def save_data(self, file_path: str):
        self.df = self.df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE, replace=False)
        self.df.to_parquet(file_path)

from typing import Tuple
import pandas as pd

from mlproject import logger
from mlproject.config.configuration import DataSplitConfig
from sklearn.model_selection import train_test_split


class DataSplit:
    def __init__(self, config: DataSplitConfig) -> None:
        self.config = config
        
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            df = pd.read_csv(self.config.data_path)
            
            X = df.drop(columns=['Status'])
            y = df['Status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            X_train.to_csv(self.config.X_train_data_path, index = False)
            X_test.to_csv(self.config.X_test_data_path, index = False)
            y_train.to_csv(self.config.y_train_data_path, index=False)
            y_test.to_csv(self.config.y_test_data_path, index=False)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise e
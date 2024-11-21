from mlproject import logger
from mlproject.components.data_split import DataSplit
from mlproject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Split Stage"
class DataSplitTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_split_config = config.get_data_split_config()
            spliter = DataSplit(data_split_config)
            X_train, X_test, y_train, y_test = spliter.split_data()
        except Exception as e:
            raise e
        
# Update main.py
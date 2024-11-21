from mlproject import logger
from mlproject.components.data_transformation import *
from mlproject.config.configuration import ConfigurationManager


STAGE_NAME ="Data Transformation Stage"
class DataTransformationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            
            # Transformed data is stored back to the data_split directory
            y_train, y_test = LabelEncoding(data_transformation_config).apply_label_encoding()
            X_train, X_test = TargetEncoding(data_transformation_config).apply_target_encoding()
            X_train, X_test = OneHotEncoding(data_transformation_config).apply_one_hot_encoding()
            X_train, X_test = FrequencyEncoding(data_transformation_config).apply_frequency_encoding()
                
        except Exception as e:
            raise e
        
# Update main.py        
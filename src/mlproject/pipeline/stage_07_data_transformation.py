from mlproject import logger
from mlproject.components.data_transformation import *
from mlproject.config.configuration import ConfigurationManager


STAGE_NAME ="Data Transformation Stage"
class DataTransformationPipeline:
    """
    Coordinates the data transformation steps for training a machine learning model.

    This class orchestrates a series of transformations applied to the dataset, including:
    - Label Encoding for the target variable
    - Target Encoding for categorical features
    - One-Hot Encoding for categorical features with two or more unique values
    - Frequency Encoding for categorical features with more than three unique values
    - Standard Scaling for numerical features
    - Handling class imbalance using ADASYN (Adaptive Synthetic Sampling)

    The class ensures the transformation is applied in the correct sequence, with each step saving 
    the transformed data to the appropriate file paths and handling any necessary preprocessing.

    Methods:
        main():
            Orchestrates the data transformation process by calling the various encoding and scaling 
            methods in sequence, applying transformations to both features and target, and handling 
            class imbalance.

    Raises:
        Exception: If any error occurs during the transformation pipeline.
    """
    def __init__(self) -> None:
        """
        Initializes the DataTransformationPipeline class.
        
        This class does not require any input arguments for initialization. The necessary configurations
        are fetched dynamically within the main method.
        """
        pass
    
    def main(self):
        """
        Executes the full data transformation pipeline.

        This method coordinates the sequence of transformations applied to the dataset, including:
        - Label Encoding (for the target variable)
        - Target Encoding (for categorical features)
        - One-Hot Encoding (for categorical features)
        - Frequency Encoding (for certain categorical features)
        - Standard Scaling (for numerical features)
        - ADASYN (to handle class imbalance)

        Each transformation step is applied in sequence, with the transformed data saved to the appropriate
        paths defined in the configuration.

        Raises:
            Exception: If any error occurs during any of the transformation steps.
        """
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            
            y_train, y_test = LabelEncoding(data_transformation_config).apply_label_encoding()
            X_train, X_test = TargetEncoding(data_transformation_config).apply_target_encoding()
            X_train, X_test = OneHotEncoding(data_transformation_config).apply_one_hot_encoding()
            X_train, X_test = FrequencyEncoding(data_transformation_config).apply_frequency_encoding()
            X_train_scaled_df, X_test_scaled_df = DataScaling(data_transformation_config).apply_standard_scaler()
            X_resampled, y_train = HandlingImbalanceDataset(data_transformation_config).apply_ADASYN(X_train_scaled_df)
            X_resampled, X_test = FeatureSelection(data_transformation_config).select_features()
    
        except Exception as e:
            raise e
        
# Update main.py        
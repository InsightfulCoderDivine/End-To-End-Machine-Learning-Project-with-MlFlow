from mlproject import logger
from mlproject.components.data_manipulation import ManipulateData
from mlproject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Manipulation Stage"

class DataManipulationTrainingPipeline:
    """
    Executes the data manipulation pipeline for preprocessing datasets.

    This class manages the entire data manipulation process by coordinating configurations 
    and invoking specific methods to handle data type corrections and value replacements.

    Methods:
        main():
            Orchestrates the data manipulation process by:
            - Loading configuration settings.
            - Initializing the `ManipulateData` class.
            - Performing data type corrections and value replacements.
    """
    
    def __init__(self) -> None:
        """
        Initializes the DataManipulationTrainingPipeline.

        Currently, no arguments are required for initialization.
        """
        pass
    
    def main(self):
        """
        Executes the main steps of the data manipulation pipeline.

        - Retrieves configuration settings for data manipulation using `ConfigurationManager`.
        - Creates a `ManipulateData` object with the retrieved configuration.
        - Performs the following operations:
          1. Corrects data types in the dataset.
          2. Updates specific values in the dataset.
        - Ensures that all operations are executed in sequence.

        Raises:
            Exception: If any error occurs during the data manipulation process.
        """
        try:
            config = ConfigurationManager()
            data_manipulation_config = config.get_data_manipulation_config()
            
            manipulator = ManipulateData(data_manipulation_config)
            manipulator.correct_dtype()
            manipulator.change_values()
            
        except Exception as e:
            raise e
        
# Update main.py
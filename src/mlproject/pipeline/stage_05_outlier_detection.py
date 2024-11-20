from mlproject import logger
from mlproject.components.outlier_detection import OutlierDetection
from mlproject.config.configuration import ConfigurationManager


STAGE_NAME = "Outler Detection Stage"
class OutlierDetectionTrainingPipeline:
    """
    Executes the pipeline for outlier detection and removal.

    This class orchestrates the entire process of outlier handling by:
    - Loading the necessary configurations.
    - Initializing the `OutlierDetection` class.
    - Performing the outlier detection and removal steps.

    Methods:
        main():
            Coordinates the execution of the outlier detection pipeline.
    """
    
    def __init__(self) -> None:
        """
        Initializes the OutlierDetectionTrainingPipeline.

        Currently, no arguments are required for initialization.
        """
        pass
    
    def main(self):
        """
        Executes the main steps of the outlier detection pipeline.

        - Retrieves configuration settings for outlier detection using `ConfigurationManager`.
        - Creates an `OutlierDetection` object with the retrieved configuration.
        - Calls the `handle_outliers` method to detect and remove outliers from the dataset.

        Raises:
            Exception: If any error occurs during the execution of the pipeline.
        """
        try:
            config = ConfigurationManager()
            outlier_detection_config = config.get_outlier_detection_config()
            outlier_handler = OutlierDetection(outlier_detection_config)
            outlier_handler.handle_outliers()
        except Exception as e:
            raise e
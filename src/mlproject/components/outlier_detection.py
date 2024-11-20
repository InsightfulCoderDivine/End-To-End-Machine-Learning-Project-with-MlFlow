import os
import pandas as pd
from mlproject import logger
from mlproject.entity.config_entity import OutlierDetectionConfig
from mlproject.utils.common import get_outliers, create_directories


class OutlierDetection:
    """
    Handles the detection and removal of outliers in a dataset.

    This class is responsible for identifying outliers in numeric columns of a dataset, 
    removing them, and saving the cleaned data to a specified directory. It also logs 
    the status of the operation.

    Attributes:
        config (OutlierDetectionConfig): Configuration object containing file paths and 
                                         directories needed for outlier detection.
    """
    
    def __init__(self, config: OutlierDetectionConfig) -> None:
        """
        Initializes the OutlierDetection object.

        Args:
            config (OutlierDetectionConfig): Configuration object containing file paths 
                                             and directories for outlier detection.
        """
        self.config = config
        
    def handle_outliers(self) -> pd.DataFrame:
        """
        Detects and removes outliers from specific numeric columns in the dataset.

        - Identifies outliers in predefined numeric columns using the `get_outliers` function.
        - Drops rows with outliers from the dataset.
        - Saves the cleaned dataset to the configured directory.
        - Logs the status of the operation in the status file.

        Returns:
            pd.DataFrame: The DataFrame after removing outliers.

        Raises:
            Exception: If any error occurs during the outlier handling process.
        """
        try:
            df = pd.read_csv(self.config.manipulated_data_dir)
            numeric_columns_with_outliers = ['loan_amount', 'income', 'Upfront_charges', 'Interest_rate_spread', 'rate_of_interest', 'property_value', 'dtir1', 'LTV']

            for col in numeric_columns_with_outliers:
                outliers = get_outliers(df, col)
                df.drop(df[outliers].index, inplace=True)
                
            # Save DataFrame without outliers
            df_without_outliers_file_path = self.config.data_without_outliers_dir  # File path for saving
            create_directories([os.path.dirname(df_without_outliers_file_path)])  # creates this directory "artifacts/outlier_detection" if it doesn't already exist.
            df.to_csv(df_without_outliers_file_path, index=False)
            
            # Log status
            with open(self.config.STATUS_FILE, "a") as status_file:
                status_file.write("Outliers handles and data saved successfully.\n")
            return df
        except Exception as e:
            raise e
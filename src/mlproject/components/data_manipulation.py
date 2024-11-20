import os
import pandas as pd
from mlproject.entity.config_entity import DataManipulationConfig
from mlproject.utils.common import create_directories
from mlproject import logger


class ManipulateData:
    """
    Handles data manipulation tasks for a dataset.

    This class is responsible for correcting data types and modifying values in a dataset 
    based on specific rules. It reads the data from configured directories, applies the 
    changes, saves the modified data, and logs the status of operations.

    Attributes:
        config (DataManipulationConfig): Configuration object containing file paths and 
                                         directories required for data manipulation.

    """
    def __init__(self, config: DataManipulationConfig) -> None:
        """
        Initializes the ManipulateData object.

        Args:
            config (DataManipulationConfig): Configuration object containing file paths and 
                                             directories required for data manipulation.
        """
        self.config = config
        
    def correct_dtype(self) -> pd.DataFrame:
        """
        Corrects the data types of specific columns in the dataset.

        - Converts the `term` column to an integer type.
        - Converts the `Status` column to an object type.
        - Saves the updated dataset to the configured directory.
        - Logs the success status to the status file.

        Returns:
            pd.DataFrame: The DataFrame with corrected data types.

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            df = pd.read_csv(self.config.cleaned_data_dir)
            
            df['term'] = df['term'].astype('int')
            df['Status'] = df['Status'].astype('object')
            
            # Save the corrected DataFrame
            corrected_df_file_path = self.config.manipulated_data_dir  # File path for saving
            create_directories([os.path.dirname(corrected_df_file_path)])  # creates this directory "artifacts/handle_missing_values" if it doesn't already exist.
            df.to_csv(corrected_df_file_path, index=False)
            
            # Log status
            with open(self.config.STATUS_FILE, "a") as status_file:
                status_file.write("Corrected dtypes and data saved successfully.\n")
            return df
        except Exception as e:
            raise e
        
    def change_values(self) -> pd.DataFrame:
        """
        Replaces specific values in the dataset based on predefined rules.

        - Fixes typos in the `Security_Type` column.
        - Replaces codes in the `occupancy_type` column with descriptive labels.
        - Saves the updated dataset to the configured directory.
        - Logs the success status to the status file.

        Returns:
            pd.DataFrame: The DataFrame with updated values.

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            df = pd.read_csv(self.config.manipulated_data_dir)
            
            df['Security_Type'] = df['Security_Type'].replace({'Indriect':'Indirect'}) 
            df['occupancy_type'] = df['occupancy_type'].replace({'pr':'Primary Residential', 'sr':'Secondary Residdential', 'ir':'Investment Residential'}) 
            
            # Save the corrected DataFrame
            corrected_df_file_path = self.config.manipulated_data_dir  # File path for saving
            create_directories([os.path.dirname(corrected_df_file_path)])  # creates this directory "artifacts/handle_missing_values" if it doesn't already exist.
            df.to_csv(corrected_df_file_path, index=False)
            
            # Log status
            with open(self.config.STATUS_FILE, "a") as status_file:
                status_file.write("Corrected values and data saved successfully.\n")
            return df
        except Exception as e:
            raise e
        
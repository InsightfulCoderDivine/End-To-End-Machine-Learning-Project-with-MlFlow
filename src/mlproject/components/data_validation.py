import pandas as pd
from typing import Tuple
from mlproject.entity.config_entity import DataValidationConfig
from mlproject import logger


class DataValidation:
    """Performs data validation checks based on the provided configuration.

    This class validates the columns and data types of a dataset against a predefined schema. 
    It checks whether all expected columns are present and whether their data types match the schema.
    
    """
    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation object with the provided configuration.

        Args:
            config (DataValidationConfig): The configuration object containing the schema 
                                          and file paths for data validation.
        """
        self.config = config
        
        
    def validate_all_columns(self) -> Tuple[bool, bool, bool]:
        """
        Validates the dataset against the predefined schema.

        This method checks:
            1. If all columns in the dataset match the schema.
            2. If the data types of the columns in the dataset match the expected types in the schema.
            3. If there are duplicate rows in the dataset.

        The validation results are written to the status file specified in the configuration.

        Returns:
            Tuple[bool, bool, bool]: A tuple containing three boolean values:
            - First value: True if all columns match the schema, False otherwise.
            - Second value: True if all data types match the schema, False otherwise.
            - Third value: True if no duplicate rows exist, False otherwise.

        Raises:
            Exception: If any error occurs during the validation process.
        """
        try:
            validation_status_col = None
            validation_status_dtype = None
            validation_status_duplicates = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_dtypes = list(data.dtypes)
            
            all_schema = self.config.all_schema.keys()
            all_schema_dtypes = self.config.all_schema.values()
            
            # Validate columns
            for col in all_cols:
                if col not in all_schema:
                    validation_status_col = False
                else:
                    validation_status_col = True

            # Validate data types
            for dtype in all_dtypes:
                if dtype not in all_schema_dtypes:
                    validation_status_dtype = False
                else:
                    validation_status_dtype = True
                    
            # Check for Duplicate rows
            if data.duplicated().any():
                # 'No duplicate rows: False'
                validation_status_duplicates = False
            else:
                # 'No duplicate rows: True'
                validation_status_duplicates = True

            # Write final results
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Columns Validation status: {validation_status_col}\n")
                f.write(f"Dtypes Validation status: {validation_status_dtype}\n")
                f.write(f"No Duplicate Rows status: {validation_status_duplicates}\n")

            return validation_status_col, validation_status_dtype, validation_status_duplicates
        except Exception as e:
            raise e
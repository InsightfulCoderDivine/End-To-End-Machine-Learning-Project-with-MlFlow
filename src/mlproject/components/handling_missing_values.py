import os
from mlproject import logger
import pandas as pd
from mlproject.config.configuration import MissingValuesConfig
from mlproject.utils.common import get_missing_columns, create_directories

class HandleMissingValues:
    def __init__(self, config: MissingValuesConfig):
        self.config = config
           
    def handle_missing_values(self) -> pd.DataFrame:
        try:
            columns_having_rows_to_drop = [
                    'loan_limit',
                    'approv_in_adv',
                    'loan_purpose',
                    'Neg_ammortization',
                    'age',
                    'submission_of_application',
                    'term']
            
            df = pd.read_csv(self.config.unzip_data_dir)
            
            # Drop rows with missing values in these columns
            df.dropna(subset=columns_having_rows_to_drop, axis=0, inplace=True)
        
            df['income'] = df.groupby('age')['income'].transform(lambda x: x.fillna(x.median()))                    
            df['property_value'] = df.groupby('Region')['property_value'].transform(lambda x: x.fillna(x.median()))
            df['rate_of_interest'] = df['rate_of_interest'].transform(lambda x: x.fillna(x.mean()))
            df['Interest_rate_spread'] = df['Interest_rate_spread'].transform(lambda x: x.fillna(x.mean()))
            df['Upfront_charges'] = df['Upfront_charges'].transform(lambda x: x.fillna(x.median()))
            df['LTV'] = df['LTV'].fillna((df['loan_amount'] / df['property_value']) * 100)
            df['dtir1'] = df['dtir1'].interpolate(method='linear')

            
            # Save the cleaned DataFrame
            cleaned_file_path = self.config.cleaned_data_dir  # File path for saving
            create_directories([os.path.dirname(cleaned_file_path)])  # creates this directory "artifacts/handle_missing_values" if it doesn't already exist.
            df.to_csv(cleaned_file_path, index=False)
            
            # Log status
            with open(self.config.STATUS_FILE, "a") as status_file:
                status_file.write("Missing values handled and data saved successfully.\n\n")
            return df
        except Exception as e:
            raise e
        
    def list_missing_values(self):
        try:            
            data_dir = self.config.cleaned_data_dir
            df = pd.read_csv(data_dir)
            # Filter columns with missing values
            numeric_columns_with_na, category_columns_with_na, _, _ = get_missing_columns(df)

            # Write final results
            with open(self.config.STATUS_FILE, "a") as f:
                f.write("Numeric Columns.\n")
                f.write(f"Columns with Missing Values: {numeric_columns_with_na}")
                f.write("\nCategorical Columns.\n")
                f.write(f"Columns with Missing Values: {category_columns_with_na}")

        except Exception as e:
            raise e
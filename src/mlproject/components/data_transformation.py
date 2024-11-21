import pandas as pd
from typing import Tuple
from mlproject import logger
from sklearn.preprocessing import LabelEncoder

from mlproject.entity.config_entity import DataTransformationConfig


class LabelEncoding:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def apply_label_encoding(self) -> Tuple[pd.Series, pd.Series]:
        try:
            y_train = pd.read_csv(self.config.y_train_data_path)
            y_test = pd.read_csv(self.config.y_test_data_path)

            # Converting first column to Series 
            y_train = y_train.iloc[:,0]
            y_test = y_test.iloc[:,0]
            
            labelEncoder = LabelEncoder()

            y_train = labelEncoder.fit_transform(y_train)
            y_test = labelEncoder.transform(y_test)

            y_train = pd.Series(y_train, name='Status')
            y_test = pd.Series(y_test, name='Status')
            
            y_train.to_csv(self.config.y_train_data_path, index=False)
            y_test.to_csv(self.config.y_test_data_path, index=False)
            
            return y_train, y_test
        except Exception as e:
            raise e
        
        
class TargetEncoding:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def apply_target_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X_train = pd.read_csv(self.config.X_train_data_path)
            X_test = pd.read_csv(self.config.X_test_data_path)
            y_train = pd.read_csv(self.config.y_train_data_path)
            target_column ='Status'
            
            # Step 1: Combine `X_train` and `y_train` for mean encoding
            security_type_df = pd.concat([X_train['Security_Type'], y_train], axis=1)

            # Step 2: Calculate mean target for each category in the training set
            security_type_mean = security_type_df.groupby('Security_Type')[target_column].mean()

            # Step 3: Map mean encoding to the training set
            X_train['Security_Type'] = X_train['Security_Type'].map(security_type_mean)

            # Step 4: Map mean encoding to the test set
            X_test['Security_Type'] = X_test['Security_Type'].map(security_type_mean)

            # Step 5: Handle categories in test set that are missing in training
            fallback_value = y_train.mean()  # Overall mean target value
            X_test['Security_Type'] = X_test['Security_Type'].fillna(fallback_value)
            
            X_train.to_csv(self.config.X_train_data_path, index = False)
            X_test.to_csv(self.config.X_test_data_path, index = False)
            
            return X_train, X_test
        except Exception as e:
            raise e

        
class OneHotEncoding:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def apply_one_hot_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            
            X_train = pd.read_csv(self.config.X_train_data_path)
            X_test = pd.read_csv(self.config.X_test_data_path)
            nunique_2_to_3 = [
                'loan_limit',
                'approv_in_adv',
                'loan_type',
                'Credit_Worthiness',
                'open_credit',
                'business_or_commercial',
                'Neg_ammortization',
                'interest_only',
                'lump_sum_payment',
                'construction_type',
                'occupancy_type',
                'Secured_by',
                'co-applicant_credit_type',
                'submission_of_application',
                'Security_Type'
                ]
            
            # Remove 'Security_Type' as it will not be one-hot encoded
            nunique_2_to_3.remove('Security_Type')

            # One-hot encode on X_train and assign back to X_train
            X_train = pd.get_dummies(X_train, columns=nunique_2_to_3)

            # One-hot encode on X_test and ensure consistent columns with X_train
            X_test = pd.get_dummies(X_test, columns=nunique_2_to_3)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            
            X_train.to_csv(self.config.X_train_data_path, index = False)
            X_test.to_csv(self.config.X_test_data_path, index = False)
            
            return X_train, X_test
        except Exception as e:
            raise e
        
        
class FrequencyEncoding:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def apply_frequency_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X_train = pd.read_csv(self.config.X_train_data_path)
            X_test = pd.read_csv(self.config.X_test_data_path)
            greater_than_3 = [
                'Gender', 'loan_purpose', 'total_units', 'credit_type', 'age', 'Region'
                ]
            
            for col in greater_than_3:
                # Step 1: Calculate frequency map from X_train
                frequency_map = X_train[col].value_counts().to_dict()
                
                # Step 2: Apply frequency map to X_train
                X_train[col] = X_train[col].map(frequency_map)
                
                # Step 3: Apply the same frequency map to X_test
                X_test[col] = X_test[col].map(frequency_map)
                
                # Step 4: Handle categories in X_test not seen in X_train
                X_test[col] = X_test[col].fillna(0)  # Replace NaN with 0
               
            # Save the transformed data back to data_split directory 
            X_train.to_csv(self.config.X_train_data_path, index = False)
            X_test.to_csv(self.config.X_test_data_path, index = False)
            
            return X_train, X_test
        except Exception as e:
            raise e                        
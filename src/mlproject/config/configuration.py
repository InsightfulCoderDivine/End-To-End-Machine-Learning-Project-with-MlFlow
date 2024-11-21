from mlproject.constants import * 
from mlproject.utils.common import read_yaml, create_directories
from mlproject.entity.config_entity import DataIngestionConfig, DataManipulationConfig, DataSplitConfig, DataTransformationConfig, DataValidationConfig, MissingValuesConfig, OutlierDetectionConfig


class ConfigurationManager:
    """
    Manages the configuration and setup for the project.

    This class is responsible for reading configuration files, creating required directories, 
    and providing specific configuration objects needed for various components of the project.

    Attributes:
        config (dict): Parsed content of the main configuration file.
        params (dict): Parsed content of the parameters file.
        schema (dict): Parsed content of the schema file.
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH, # file path from 'constants' folder 
        params_filepath = PARAMS_FILE_PATH, # file path from 'constants' folder
        schema_filepath = SCHEMA_FILE_PATH # file path from 'constants' folder
        ):
        """
        Initializes the ConfigurationManager.

        Reads YAML configuration files for main configuration, parameters, and schema. 
        Also ensures that the root artifacts directory specified in the configuration is created.

        Args:
            config_filepath (str): Path to the main configuration YAML file. Default is `CONFIG_FILE_PATH`.
            params_filepath (str): Path to the parameters YAML file. Default is `PARAMS_FILE_PATH`.
            schema_filepath (str): Path to the schema YAML file. Default is `SCHEMA_FILE_PATH`.
        """
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        # Creates 'artifacts' folder
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Function to return data_ingestion directories.

        Returns:
            DataIngestionConfig: DataIngestionConfig type.
        """
        # Access all 'data_ingestion' variables from config.yaml
        config = self.config.data_ingestion
        
        # Creates data_ingestion directory
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_validation_config(self) -> DataValidationConfig:
        """
        Creates and returns a `DataValidationConfig` object for data validation.

        This method retrieves the data validation-specific configuration from the main 
        configuration and schema files. It also ensures the directories required for 
        data validation are created.

        Returns:
            DataValidationConfig: An instance of `DataValidationConfig` initialized with 
            the appropriate paths and schema information.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_data_dir=config.unzip_data_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        return data_validation_config
    
    def get_missing_values_config(self) -> MissingValuesConfig:
        """
        Creates and returns a `MissingValuesConfig` object for handling missing values.

        This method retrieves the configuration details specific to handling missing values from 
        the main configuration file. It also ensures that the directories required for this step 
        are created.

        Returns:
            MissingValuesConfig: An instance of `MissingValuesConfig` containing paths 
            and other configuration details for handling missing values.

        Raises:
            Exception: If an error occurs while retrieving the missing values configuration.
        """
        try:
            config = self.config.handle_missing_values
            create_directories([config.root_dir])
            
            missing_values_config = MissingValuesConfig(
                root_dir=config.root_dir,
                unzip_data_dir=config.unzip_data_dir,
                cleaned_data_dir=config.cleaned_data_dir,
                STATUS_FILE=config.STATUS_FILE,
            )
            return missing_values_config
        except Exception as e:
            raise e
        
    def get_data_manipulation_config(self) -> DataManipulationConfig:
        """
        Creates and returns a `DataManipulationConfig` object for manipulating data.

        This method retrieves the configuration details specific to manipulating data values from 
        the main configuration file. It also ensures that the directories required for this step 
        are created.

        Returns:
            DataManipulationConfig: An instance of `DataManipulationConfig` containing paths 
            and other configuration details for manipulating data.

        Raises:
            Exception: If an error occurs while retrieving the manipulating data configuration.
        """
        try:
            config = self.config.data_manipulation
            create_directories([config.root_dir])
            
            data_manipulation_config = DataManipulationConfig(
                root_dir=config.root_dir,
                cleaned_data_dir=config.cleaned_data_dir,
                manipulated_data_dir=config.manipulated_data_dir,
                STATUS_FILE=config.STATUS_FILE
            )
            return data_manipulation_config
        except Exception as e:
            raise e
        
    def get_outlier_detection_config(self) -> OutlierDetectionConfig:
        try:
            config = self.config.outlier_detection
            create_directories([config.root_dir])
            
            outlier_detection_config = OutlierDetectionConfig(
                root_dir=config.root_dir,
                manipulated_data_dir=config.manipulated_data_dir,
                data_without_outliers_dir=config.data_without_outliers_dir,
                STATUS_FILE=config.STATUS_FILE
            )
            return outlier_detection_config
        except Exception as e:
            raise e
        
    def get_data_split_config(self) -> DataSplitConfig:
        try:
            config = self.config.data_split
            create_directories([config.root_dir])
            
            data_split_config = DataSplitConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
                X_train_data_path=config.X_train_data_path,
                X_test_data_path=config.X_test_data_path,
                y_train_data_path=config.y_train_data_path,
                y_test_data_path=config.y_test_data_path
            )
            return data_split_config
        except Exception as e:
            raise e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config.data_transformation
            # Not necessary because we are not saving any file 
            # All transformed data would be saved back to `data_split` directory
            # create_directories([config.root_dir])
            
            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                X_train_data_path=config.X_train_data_path,
                X_test_data_path=config.X_test_data_path,
                y_train_data_path=config.y_train_data_path,
                y_test_data_path=config.y_test_data_path
            )
            return data_transformation_config
        except Exception as e:
            raise e
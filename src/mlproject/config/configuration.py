from mlproject.constants import * 
from mlproject.utils.common import read_yaml, create_directories
from mlproject.entity.config_entity import DataIngestionConfig, DataValidationConfig


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
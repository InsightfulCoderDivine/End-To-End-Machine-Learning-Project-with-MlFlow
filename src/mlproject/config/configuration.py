from mlproject.constants import * 
from mlproject.utils.common import read_yaml, create_directories
from mlproject.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH, # file path from 'constants' folder 
        params_filepath = PARAMS_FILE_PATH, # file path from 'constants' folder
        schema_filepath = SCHEMA_FILE_PATH # file path from 'constants' folder
        ):
        
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
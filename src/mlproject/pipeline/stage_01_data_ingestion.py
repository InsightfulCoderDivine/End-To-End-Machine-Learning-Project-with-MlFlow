from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_ingestion import DataIngestion

from mlproject import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        """Method that runs the data ingestion pipeline.
        """
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            
        except Exception as e:
            raise e
        
        
# Run the main method
# - To test pipeline add it to `main.py`
if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)        
        raise e
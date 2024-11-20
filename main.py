from mlproject import logger
from mlproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlproject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlproject.pipeline.stage_03_handling_missing_values import HandlingMissingValuesTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
    
except Exception as e:
    logger.exception(e)        
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
        
except Exception as e:
    logger.exception(e)        
    raise e    


STAGE_NAME = 'Handling Missing Values Stage'
try:
    logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
    handle_missing_values = HandlingMissingValuesTrainingPipeline()
    handle_missing_values.main()
    logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
    
except Exception as e:
    logger.exception(e)        
    raise e  
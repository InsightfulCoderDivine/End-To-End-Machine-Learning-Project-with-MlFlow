from mlproject import logger
from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_validation import DataValidation


STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_validation_config()
            data_validation = DataValidation(data_validation_config)
            data_validation.validate_all_columns()
        except Exception as e:
            raise e


# 8. Update the main.py
# if __name__ == "__main__":
#         try:
#             logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
#             obj = DataValidationTrainingPipeline()
#             obj.main()
#             logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
            
#         except Exception as e:
#             logger.exception(e)        
#             raise e    
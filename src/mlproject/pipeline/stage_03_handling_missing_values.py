from mlproject import logger
from mlproject.components.handling_missing_values import HandleMissingValues
from mlproject.config.configuration import ConfigurationManager

STAGE_NAME = 'Handling Missing Values Stage'

class HandlingMissingValuesTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()
            
            # Get missing values configuration
            missing_values_config = config.get_missing_values_config()
            
            # Handle missing values
            missing_values_handler = HandleMissingValues(missing_values_config)
            cleaned_df = missing_values_handler.handle_missing_values()
            
            # List missing values after handling
            missing_values_handler.list_missing_values()
            
            
        except Exception as e:
            raise e
        
# # To test: 
# # Update main.py
# # delete arifacts folder
# # open terminal and run main.py

# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
#         handle_missing_values = HandlingMissingValuesTrainingPipeline()
#         handle_missing_values.main()
#         logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
        
#     except Exception as e:
#         logger.exception(e)        
#         raise e  
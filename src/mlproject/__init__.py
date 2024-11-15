# logging file
import os
import sys
import logging

# Format:= [time: linenum mlprojectLogger: INFO: file: mssg]
logging_str = "[%(asctime)s: %(lineno)d %(name)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format=logging_str, 
    handlers=[
        logging.FileHandler(log_filepath), # creates the log folder, inside that it would save all the logging to log_filepath
        logging.StreamHandler(sys.stdout)  # prints your log in your terminal
    ]
)
# Initialize logging
logger = logging.getLogger("mlprojectLogger")
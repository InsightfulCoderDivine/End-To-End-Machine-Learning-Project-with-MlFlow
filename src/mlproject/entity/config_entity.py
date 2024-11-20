from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration class for data validation.

    This dataclass stores the configuration details required for validating data.
    The `frozen=True` parameter ensures that instances of this class are immutable.

    Attributes:
        root_dir (Path): The root directory where data and related files are stored.
        unzip_data_dir (Path): Directory where the unzipped data files are located.
        STATUS_FILE (str): Name or path of the file that tracks the status of the data validation process.
        all_schema (dict): Dictionary containing the schema details for data validation.
                           Typically includes keys and expected data types for the datasets.
    """
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema: dict
    

@dataclass(frozen=True)
class MissingValuesConfig:
    """
    Configuration class for handling missing values in a dataset.

    This dataclass stores the configuration details required for identifying and managing 
    missing values during data preprocessing. The `frozen=True` parameter ensures that 
    instances of this class are immutable.

    Attributes:
        root_dir (Path): The root directory where all related files and data are stored.
        unzip_data_dir (Path): Directory containing the unzipped raw data files.
        cleaned_data_dir (str): Directory where the cleaned data files will be stored 
                                after handling missing values.
        STATUS_FILE (str): Path to the file where the status of missing value handling 
                           will be logged.
    """
    root_dir: Path
    unzip_data_dir: Path
    cleaned_data_dir: str
    STATUS_FILE: str
    
    
@dataclass
class DataManipulationConfig:
    """
    Configuration class for manipulating data in a dataset.

    This dataclass stores the configuration details required for manipulating data
    during data preprocessing. The `frozen=True` parameter ensures that 
    instances of this class are immutable.

    Attributes:
        root_dir (Path): The root directory where all related files and data are stored.
        cleaned_data_dir (str): Directory where the cleaned data files is stored 
                                after handling missing values.
        manipulated_data_dir (str): Directory where the manipulated data files will be stored
        STATUS_FILE (str): Path to the file where the status of data manipulation 
                           will be logged.
    """
    root_dir: Path
    cleaned_data_dir: str
    manipulated_data_dir: str
    STATUS_FILE: str


@dataclass(frozen=True)
class OutlierDetectionConfig:
    """
    Configuration class for outlier detection in a dataset.

    This dataclass contains the file paths and directories needed for identifying and 
    handling outliers during data preprocessing. The `frozen=True` parameter ensures 
    immutability of the configuration object.

    Attributes:
        root_dir (Path): The root directory for storing all related files and outputs.
        manipulated_data_dir (str): Path to the directory containing the data after initial manipulations.
        data_without_outliers_dir (str): Path to the directory where data without outliers will be saved.
        STATUS_FILE (str): Path to the file for logging the status of outlier detection operations.
    """
    root_dir: Path
    manipulated_data_dir: str
    data_without_outliers_dir: str
    STATUS_FILE: str
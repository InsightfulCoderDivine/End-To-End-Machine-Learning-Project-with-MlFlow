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
    
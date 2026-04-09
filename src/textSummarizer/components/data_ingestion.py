import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file) or os.path.getsize(self.config.local_data_file) == 0:
            logger.info("Downloading file using curl...")
            os.system(f"curl -L -o {self.config.local_data_file} {self.config.source_URL}")
            logger.info(f"{self.config.local_data_file} downloaded successfully!")
        else:
            logger.info(f"File already exists")  

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

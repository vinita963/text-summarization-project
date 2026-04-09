from textSummarizer.logging import logger
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.components.data_validation import DataValidation
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    logger.info(">>>>> Starting MLOps Pipeline Execute <<<<<")
    config_manager = ConfigurationManager()

    # Stage 1: Ingestion
    logger.info(">>>>>>>> stage 01: Data Ingestion started <<<<<<<<")
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()

    # Stage 2: Validation
    logger.info(">>>>>>>> stage 02: Data Validation started <<<<<<<<")
    data_validation_config = config_manager.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_files_exist()

    # Stage 3: Transformation
    logger.info(">>>>>>>> stage 03: Data Transformation started <<<<<<<<")
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.convert()

    # Stage 4: Generative Fine-Tuning
    logger.info(">>>>>>>> stage 04: Model Trainer started <<<<<<<<")
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainer(config=model_trainer_config)
    model_trainer.train()
    logger.info(">>>>> Pipeline Execute Completed Successfully! <<<<<")
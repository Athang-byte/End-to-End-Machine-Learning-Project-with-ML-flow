import MLproject.config.configuration as cfg
print("Using config file from:", cfg.__file__)

from MLproject import logger
from MLproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from MLproject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from MLproject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from MLproject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline


STAGE_NAME = "Data Injestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
except Exception as e:
        logger.exception(e)
        raise e


from MLproject.components.data_validation import DataValidation
from MLproject.config.configuration import ConfigurationManager
from MLproject import logger



STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()

    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_columns()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME= "Data Transformation Stage"

try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion=DataTransformationTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training Stage "


try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj=ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
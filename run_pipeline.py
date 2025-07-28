# run_pipeline.py
import os
project_root = os.path.dirname(os.path.abspath(__file__))  # full path to run_pipeline.py
config_path = os.path.join(project_root, "src", "config", "config.yaml")

from src.pipeline.extract import DataExtractor
from src.pipeline.transform import DataTransformer
from src.pipeline.load import DataLoader
from src.utils.logger import get_logger
from src.utils.exceptions import ProjectBaseError

logger = get_logger(__name__)

def run_etl_pipeline():
    try:
        logger.info("Starting ETL pipeline...")

        # Step 1: Extract
        extractor = DataExtractor(config_path)
        raw_data = extractor.extract_raw_data()

        # Step 2: Transform
        transformer = DataTransformer(config_path)
        cleaned_data = transformer.transform(raw_data)
        logger.info(f"Transformed data shape: {cleaned_data.shape}")


        # Step 3: Load
        loader = DataLoader(config_path)
        loader.load_data(cleaned_data)

        logger.info("ETL pipeline completed successfully!")

    except ProjectBaseError as e:
        logger.error(f"ETL pipeline failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error occurred during ETL pipeline.")

if __name__ == "__main__":
    run_etl_pipeline()

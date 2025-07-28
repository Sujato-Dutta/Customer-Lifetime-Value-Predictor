# run_preprocessor.py

from run_feature_engineering import run_feature_engineering
from src.preprocessor import PreprocessorBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_preprocessor():
    logger.info("Starting preprocessing pipeline...")

    # Get feature-engineered data
    df = run_feature_engineering()  # Reuse logic

    # Build and save the preprocessor
    preprocessor = PreprocessorBuilder()
    preprocessor.build_and_save(df)

    logger.info("Preprocessor built and saved successfully.")

if __name__ == "__main__":
    run_preprocessor()

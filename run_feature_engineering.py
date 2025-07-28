# run_feature_engineering.py

import os
from src.data_ingestion import SupabaseIngestor
from src.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_feature_engineering():
    logger.info("Starting data ingestion & feature engineering pipeline...")

    # Load data from Supabase
    config_path = os.path.join("src", "config", "config.yaml")
    ingestor = SupabaseIngestor(config_path)
    df = ingestor.load_data()

    logger.info(f"Loaded {df.shape[0]} records. Starting feature engineering...")

    # Perform feature engineering
    fe = FeatureEngineer()
    df = fe.add_clv_feature(df)  # Adds the target column 'customer_lifetime_value'

    logger.info("Feature engineering completed successfully.")

    return df  # Pass it directly to the next step (e.g., preprocessor or training)

if __name__ == "__main__":
    run_feature_engineering()

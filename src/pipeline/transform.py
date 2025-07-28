# src/pipeline/transform.py

import pandas as pd
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError
from src.config.config_loader import load_config

logger = get_logger(__name__)

class DataTransformer:
    """
    Applies transformations to the raw customer transactions data.
    """

    def __init__(self,config_path: str):
        try:
            self.config = load_config(config_path)
            self.logger = get_logger(__name__)
            self.transformed_table_name = self.config['supabase']['transformed_table_name']
        except Exception as e:
            raise ProjectBaseError(f"Error loading config in transformer: {e}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the raw data for modeling:
        - Converts date columns
        - Handles missing values
        - Standardizes column types

        :param df: Raw input DataFrame
        :return: Transformed DataFrame
        """
        try:
            logger.info("Starting data transformation...")

            # Ensure required columns exist
            required_columns = [
                "customer_id", "invoice_id", "invoice_date",
                "purchase_amount", "product_category",
                "payment_method", "customer_segment", "region"
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ProjectBaseError(f"Missing columns in raw data: {missing_cols}")

            # Convert invoice_date to datetime
            df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

            # Drop rows with missing critical fields
            df = df.dropna(subset=["customer_id", "invoice_date", "purchase_amount"])

            # Cast purchase_amount to float
            df["purchase_amount"] = df["purchase_amount"].astype(float)

            # Optional: Normalize text fields (e.g., title case or lower case)
            text_cols = ["product_category", "payment_method", "customer_segment", "region"]
            for col in text_cols:
                df[col] = df[col].astype(str).str.strip().str.lower()

            # Convert datetime columns to ISO string format for JSON serialization
            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
               df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"Transformation completed. Final shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise ProjectBaseError(f"Transformation failed: {e}")

# src/feature_engineering.py

import pandas as pd
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass

    def add_clv_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'customer_lifetime_value' column as total purchase amount per customer.
        """
        try:
            logger.info("Starting feature engineering: Calculating CLV.")

            clv_df = df.groupby('customer_id')['purchase_amount'].sum().reset_index()
            clv_df.rename(columns={'purchase_amount': 'customer_lifetime_value'}, inplace=True)

            df = df.merge(clv_df, on='customer_id', how='left')

            logger.info("Feature 'customer_lifetime_value' added successfully.")
            return df

        except Exception as e:
            logger.error("CLV calculation failed.")
            raise ProjectBaseError("CLV calculation failed.") from e

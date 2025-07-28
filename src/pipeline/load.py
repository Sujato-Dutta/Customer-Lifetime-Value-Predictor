# src/pipeline/load.py

import pandas as pd
from supabase import create_client, Client
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError
from src.config.config_loader import load_config

logger = get_logger(__name__)

class DataLoader:
    """
    Loads the transformed customer data into the cleaned Supabase table.
    """

    def __init__(self, config_path=None):
        config = load_config(config_path)
        supabase_url = config["supabase"]["url"]
        supabase_key = config["supabase"]["key"]
        self.table_name = config["supabase"]["transformed_table_name"]

        self.client: Client = create_client(supabase_url, supabase_key)
        logger.info(f"Connected to Supabase for loading: {supabase_url}")

    def load_data(self, df: pd.DataFrame):
        """
        Uploads transformed data to the cleaned table in Supabase.

        :param df: Transformed DataFrame ready for modeling
        """
        try:
            logger.info(f"Uploading {len(df)} rows to table: {self.table_name}")

            # Delete existing data (if overwrite logic is desired)
            self.client.table(self.table_name).delete().neq("customer_id", "").execute()

            # Upload in batches (to avoid Supabase limits)
            batch_size = 500
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size].to_dict(orient="records")
                self.client.table(self.table_name).insert(batch).execute()

            logger.info("Data loading complete.")

        except Exception as e:
            logger.error(f"Failed to load data into Supabase: {e}")
            raise ProjectBaseError(f"Loading failed: {e}")

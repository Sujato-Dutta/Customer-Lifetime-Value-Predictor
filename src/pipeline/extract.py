# src/pipeline/extract.py

import pandas as pd
from supabase import create_client, Client
from src.config.config_loader import load_config
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError

logger = get_logger(__name__)

class DataExtractor:
    def __init__(self, config_path=None):
        config = load_config(config_path)
        supabase_url = config["supabase"]["url"]
        supabase_key = config["supabase"]["key"]
        self.table_name = config["supabase"]["raw_table_name"]

        self.client: Client = create_client(supabase_url, supabase_key)
        logger.info(f"Connected to Supabase: {supabase_url}")
        
    def extract_raw_data(self) -> pd.DataFrame:
        """
        Extracts all raw customer transaction data from Supabase using pagination.
        :return: DataFrame of raw customer data
        """
        try:
            logger.info(f"Fetching raw data from table: {self.table_name}")
            all_data = []
            batch_size = 1000
            offset = 0

            while True:
                response = (
                    self.client
                    .table(self.table_name)
                    .select("*")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                batch = response.data
                if not batch:
                    break

                all_data.extend(batch)
                offset += batch_size
                logger.info(f"Fetched {len(batch)} rows... Total so far: {len(all_data)}")

            if not all_data:
                raise ProjectBaseError(f"No data found in table '{self.table_name}'")

            df = pd.DataFrame(all_data)
            logger.info(f"Completed extraction. Total rows fetched: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Failed to extract raw data: {e}")
            raise ProjectBaseError(f"Extraction failed: {e}")


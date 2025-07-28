import pandas as pd
from supabase import create_client, Client
from config.config_loader import load_config
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError

logger = get_logger(__name__)

class SupabaseIngestor:
    def __init__(self, config_path=None):
        config = load_config(config_path)
        sb_config = config['supabase']

        self.supabase_url = sb_config['url']
        self.supabase_key = sb_config['key']
        self.transformed_table_name = sb_config['transformed_table_name']

        self.client: Client = create_client(self.supabase_url, self.supabase_key)

    def load_data(self, batch_size: int = 1000) -> pd.DataFrame:
        try:
            logger.info(f"Fetching data from Supabase table: {self.transformed_table_name}")
            all_data = []
            offset = 0

            while True:
                response = (
                    self.client.table(self.transformed_table_name)
                    .select("*")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                data_batch = response.data
                if not data_batch:
                    break

                all_data.extend(data_batch)
                logger.info(f"Fetched batch with {len(data_batch)} records (offset: {offset})")
                offset += batch_size

            if not all_data:
                logger.warning("No data returned from Supabase.")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)
            logger.info(f"Ingested total of {len(df)} records from Supabase.")
            return df

        except Exception as e:
            logger.error("Failed to ingest data from Supabase.")
            raise ProjectBaseError("Supabase ingestion failed.") from e


if __name__ == "__main__":
    config_path = "src/config/config.yaml"  # Update if different
    ingestor = SupabaseIngestor(config_path=config_path)

    try:
        df = ingestor.load_data()
        print(f"Ingested DataFrame with shape: {df.shape}")
        print(df.head())
    except ProjectBaseError as e:
        print(f"Ingestion failed: {e}")

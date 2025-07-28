# src/preprocessor.py

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.logger import get_logger
from utils.exceptions import ProjectBaseError

logger = get_logger(__name__)

class PreprocessorBuilder:
    def __init__(self, artifact_path: str = "artifacts/preprocessor.pkl"):
        self.artifact_path = artifact_path

    def build_and_save(self, df: pd.DataFrame):
        """
        Builds the preprocessor and saves it to disk.
        """
        try:
            logger.info("Building preprocessor...")

            numeric_features = ['purchase_amount']
            categorical_features = ['product_category', 'payment_method', 'customer_segment', 'region']

            preprocessor = ColumnTransformer(transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

            # Only fit on features (no target leakage)
            X = df[numeric_features + categorical_features]
            preprocessor.fit(X)

            os.makedirs(os.path.dirname(self.artifact_path), exist_ok=True)
            joblib.dump(preprocessor, self.artifact_path)

            logger.info(f"Preprocessor saved to: {self.artifact_path}")

        except Exception as e:
            logger.error("Failed to build and save preprocessor.")
            raise ProjectBaseError("Preprocessor build/save failed.") from e

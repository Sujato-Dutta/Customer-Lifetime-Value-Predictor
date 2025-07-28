# model_predict.py

import os
import joblib
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

def load_preprocessor(path=PREPROCESSOR_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found at {path}")
    return joblib.load(path)

def preprocess_input(data: pd.DataFrame, preprocessor):
    logger.info("Preprocessing input data...")
    return preprocessor.transform(data)

def predict_clv(input_data: pd.DataFrame):
    logger.info("Loading model and preprocessor...")
    model = load_model()
    preprocessor = load_preprocessor()

    X_processed = preprocess_input(input_data, preprocessor)
    prediction = model.predict(X_processed)

    return prediction

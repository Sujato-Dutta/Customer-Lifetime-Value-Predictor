# run_model_predict.py

import pandas as pd
from src.model_predict import predict_clv
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_model_prediction():
    logger.info("Running CLV prediction...")

    #Example input - replace with dynamic or real-time data source as needed
    input_data = pd.DataFrame([{
        "purchase_amount": 120.0,
        "product_category": "electronics",
        "payment_method": "credit card",
        "customer_segment": "loyal",
        "region": "north"
    }])

    try:
        predicted_clv = predict_clv(input_data)
        logger.info(f"Predicted Customer Lifetime Value: {predicted_clv[0]:.2f}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    run_model_prediction()

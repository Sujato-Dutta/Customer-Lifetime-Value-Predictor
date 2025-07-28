# model_train.py

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from src.monitoring.mlflow_helper import init_mlflow_tracking, log_model_with_metrics

from run_feature_engineering import run_feature_engineering
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_preprocessor(path="artifacts/preprocessor.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found at {path}")
    return joblib.load(path)

def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    logger.info(f"Best hyperparameters: {search.best_params_}")
    return search.best_estimator_

def run_model_training():
    logger.info("Starting model training...")
    init_mlflow_tracking()  # Local or remote

    df = run_feature_engineering()
    logger.info(f"Data shape: {df.shape}")

    X = df.drop(["customer_lifetime_value", "invoice_date", "invoice_id", "customer_id"], axis=1)
    y = df["customer_lifetime_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = load_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info("Data transformed. Starting model training...")
    best_model = train_model(X_train_processed, y_train)

    y_pred = best_model.predict(X_test_processed)
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "r2": r2}
    params = best_model.get_params()

    log_model_with_metrics(best_model, params, metrics)

    logger.info(f"RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, os.path.join("artifacts", "model.pkl"))
    logger.info("Model saved to artifacts/model.pkl")

    return best_model, params, metrics

if __name__ == "__main__":
    run_model_training()

import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

def init_mlflow_tracking(
    tracking_uri: str = "https://dagshub.com/Sujato-Dutta/Customer-Lifetime-Value-Predictor.mlflow",
    experiment_name: str = "CLV_Prediction"
):
    # Set credentials from .env
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_with_metrics(model, params, metrics, model_name: str = "RandomForestRegressor"):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

import mlflow
import mlflow.sklearn

def init_mlflow_tracking(tracking_uri: str = "http://127.0.0.1:5000", experiment_name: str = "CLV_Prediction"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_with_metrics(model, params, metrics, model_name: str = "RandomForestRegressor"):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=model_name)

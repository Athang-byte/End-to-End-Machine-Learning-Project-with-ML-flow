import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from MLproject.entity.config_entity import ModelEvaluationConfig
from MLproject.utils.common import save_json
from pathlib import Path

# ✅ Set tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/Athang-byte/End-to-End-Machine-Learning-Project-with-ML-flow.mlflow")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # Load data & model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # ✅ Set registry + experiment
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_experiment("ElasticNetModel")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Predictions
            predicted_qualities = model.predict(test_x)

            # Metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

           
            mlflow.log_param("alpha", self.config.all_params.alpha)
            mlflow.log_param("l1_ratio", self.config.all_params.l1_ratio)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # ✅ Register model correctly
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name="ElasticNetModel"  # EXACT name
                )
            else:
                mlflow.sklearn.log_model(model, "model")
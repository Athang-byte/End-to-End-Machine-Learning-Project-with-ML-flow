import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from MLproject import logger
from MLproject.entity.config_entity import ModelEvaluationConfig
from MLproject.utils.common import save_json


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    # ============================================================
    # 1. CALCULATE EVALUATION METRICS
    # ============================================================

    def eval_metrics(self, actual, pred):

        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        mae = mean_absolute_error(
            actual,
            pred
        )

        r2 = r2_score(
            actual,
            pred
        )

        return rmse, mae, r2

    # ============================================================
    # 2. CREATE ACTUAL VS PREDICTED PLOT
    # ============================================================

    def create_actual_vs_predicted_plot(
        self,
        actual,
        predicted,
        output_path
    ):

        plt.figure(figsize=(8, 6))

        plt.scatter(
            actual,
            predicted,
            alpha=0.6
        )

        # Perfect prediction line
        min_value = min(
            actual.min(),
            predicted.min()
        )

        max_value = max(
            actual.max(),
            predicted.max()
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        plt.xlabel("Actual Quality")
        plt.ylabel("Predicted Quality")
        plt.title("Actual vs Predicted Wine Quality")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

    # ============================================================
    # 3. CREATE RESIDUAL PLOT
    # ============================================================

    def create_residual_plot(
        self,
        actual,
        predicted,
        output_path
    ):

        residuals = actual - predicted

        plt.figure(figsize=(8, 6))

        plt.scatter(
            predicted,
            residuals,
            alpha=0.6
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel("Predicted Quality")
        plt.ylabel("Residual")
        plt.title("Residual Analysis")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

    # ============================================================
    # 4. CREATE FEATURE IMPORTANCE PLOT
    # ============================================================

    def create_feature_importance_plot(
        self,
        model,
        feature_names,
        output_path
    ):

        if not hasattr(
            model,
            "feature_importances_"
        ):

            logger.warning(
                "Model does not support feature importance."
            )

            return

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=True
        )

        plt.figure(figsize=(10, 7))

        plt.barh(
            importance_df["feature"],
            importance_df["importance"]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

    # ============================================================
    # 5. MAIN EVALUATION
    # ============================================================

    def log_into_mlflow(self):

        logger.info(
            "Starting model evaluation..."
        )

        # ========================================================
        # Load test data
        # ========================================================

        test_data = pd.read_csv(
            self.config.test_data_path
        )

        logger.info(
            "Test data loaded successfully."
        )

        # ========================================================
        # Load final model
        # ========================================================

        model = joblib.load(
            self.config.model_path
        )

        logger.info(
            f"Model loaded from: "
            f"{self.config.model_path}"
        )

        # ========================================================
        # Split features and target
        # ========================================================

        test_x = test_data.drop(
            [self.config.target_column],
            axis=1
        )

        test_y = test_data[
            self.config.target_column
        ]

        # ========================================================
        # Create predictions
        # ========================================================

        predicted_qualities = model.predict(
            test_x
        )

        # ========================================================
        # Calculate metrics
        # ========================================================

        rmse, mae, r2 = self.eval_metrics(
            test_y,
            predicted_qualities
        )

        logger.info(
            f"Evaluation RMSE: {rmse:.4f}"
        )

        logger.info(
            f"Evaluation MAE: {mae:.4f}"
        )

        logger.info(
            f"Evaluation R2: {r2:.4f}"
        )

        # ========================================================
        # Create evaluation directory
        # ========================================================

        evaluation_dir = Path(
            "artifacts/model_evaluation"
        )

        evaluation_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ========================================================
        # Save metrics locally
        # ========================================================

        scores = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

        save_json(
            path=Path(
                self.config.metric_file_name
            ),
            data=scores
        )

        logger.info(
            "Evaluation metrics saved successfully."
        )

        # ========================================================
        # Create plots
        # ========================================================

        actual_vs_predicted_path = (
            evaluation_dir /
            "actual_vs_predicted.png"
        )

        residual_plot_path = (
            evaluation_dir /
            "residual_plot.png"
        )

        feature_importance_path = (
            evaluation_dir /
            "feature_importance.png"
        )

        # Actual vs Predicted
        self.create_actual_vs_predicted_plot(
            test_y,
            predicted_qualities,
            actual_vs_predicted_path
        )

        # Residual plot
        self.create_residual_plot(
            test_y,
            predicted_qualities,
            residual_plot_path
        )

        # Feature importance
        self.create_feature_importance_plot(
            model,
            test_x.columns,
            feature_importance_path
        )

        logger.info(
            "Evaluation plots created successfully."
        )

        # ========================================================
        # MLflow configuration
        # ========================================================

        mlflow.set_registry_uri(
            self.config.mlflow_uri
        )

        mlflow.set_experiment(
            "WineQualityModelComparison"
        )

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        # ========================================================
        # MLflow run
        # ========================================================

        with mlflow.start_run(
            run_name="final_model_evaluation"
        ):

            # ----------------------------------------------------
            # Model information
            # ----------------------------------------------------

            mlflow.log_param(
                "model_type",
                type(model).__name__
            )

            mlflow.log_param(
                "target_column",
                self.config.target_column
            )

            mlflow.log_param(
                "evaluation_stage",
                "final_model_evaluation"
            )

            # ----------------------------------------------------
            # Log model parameters
            # ----------------------------------------------------

            if hasattr(
                model,
                "get_params"
            ):

                model_params = model.get_params()

                for param_name, param_value in model_params.items():

                    try:
                        mlflow.log_param(
                            param_name,
                            param_value
                        )
                    except Exception:
                        pass

            # ----------------------------------------------------
            # Log metrics
            # ----------------------------------------------------

            mlflow.log_metric(
                "rmse",
                float(rmse)
            )

            mlflow.log_metric(
                "mae",
                float(mae)
            )

            mlflow.log_metric(
                "r2",
                float(r2)
            )

            # ----------------------------------------------------
            # Log evaluation plots
            # ----------------------------------------------------

            mlflow.log_artifact(
                str(actual_vs_predicted_path)
            )

            mlflow.log_artifact(
                str(residual_plot_path)
            )

            if feature_importance_path.exists():

                mlflow.log_artifact(
                    str(feature_importance_path)
                )

            # ----------------------------------------------------
            # Log metrics JSON
            # ----------------------------------------------------

            mlflow.log_artifact(
                str(self.config.metric_file_name)
            )

            # ----------------------------------------------------
            # Log final model
            # ----------------------------------------------------

            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(
                    model,
                    "model"
                )

            else:

                mlflow.sklearn.log_model(
                    model,
                    "model"
                )

        logger.info(
            "Model evaluation completed successfully."
        )
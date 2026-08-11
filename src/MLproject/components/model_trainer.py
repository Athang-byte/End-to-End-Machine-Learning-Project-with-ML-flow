
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from MLproject import logger
from MLproject.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        # ============================================================
        # 1. LOAD TRAINING AND TEST DATA
        # ============================================================

        train_data = pd.read_csv(
            self.config.train_data_path
        )

        test_data = pd.read_csv(
            self.config.test_data_path
        )

        logger.info(
            "Training data loaded successfully."
        )

        logger.info(
            f"Training samples: {len(train_data)}"
        )

        logger.info(
            f"Test samples: {len(test_data)}"
        )

        # ============================================================
        # 2. SPLIT FEATURES AND TARGET
        # ============================================================

        train_x = train_data.drop(
            [self.config.target_column],
            axis=1
        )

        train_y = train_data[
            self.config.target_column
        ]

        test_x = test_data.drop(
            [self.config.target_column],
            axis=1
        )

        test_y = test_data[
            self.config.target_column
        ]

        # ============================================================
        # 3. SET UP MLFLOW
        # ============================================================

        mlflow.set_experiment(
            "WineQualityModelComparison"
        )

        # ============================================================
        # 4. CREATE MODELS
        # ============================================================

        models = {

            "elasticnet": ElasticNet(
                alpha=self.config.models[
                    "elasticnet"
                ]["alpha"],

                l1_ratio=self.config.models[
                    "elasticnet"
                ]["l1_ratio"],

                random_state=42
            ),

            "random_forest": RandomForestRegressor(
                n_estimators=self.config.models[
                    "random_forest"
                ]["n_estimators"],

                max_depth=self.config.models[
                    "random_forest"
                ]["max_depth"],

                min_samples_split=self.config.models[
                    "random_forest"
                ]["min_samples_split"],

                random_state=self.config.models[
                    "random_forest"
                ]["random_state"],

                n_jobs=-1
            ),

            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=self.config.models[
                    "gradient_boosting"
                ]["n_estimators"],

                learning_rate=self.config.models[
                    "gradient_boosting"
                ]["learning_rate"],

                max_depth=self.config.models[
                    "gradient_boosting"
                ]["max_depth"],

                random_state=self.config.models[
                    "gradient_boosting"
                ]["random_state"]
            )
        }

        # ============================================================
        # 5. CROSS-VALIDATION + MODEL EVALUATION
        # ============================================================

        results = {}

        for model_name, model in models.items():

            logger.info(
                f"Training {model_name}..."
            )

            with mlflow.start_run(
                run_name=model_name
            ):

                # ----------------------------------------------------
                # MLflow: Log model name
                # ----------------------------------------------------

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                # ----------------------------------------------------
                # Log model hyperparameters
                # ----------------------------------------------------

                if model_name == "elasticnet":

                    mlflow.log_param(
                        "alpha",
                        self.config.models[
                            "elasticnet"
                        ]["alpha"]
                    )

                    mlflow.log_param(
                        "l1_ratio",
                        self.config.models[
                            "elasticnet"
                        ]["l1_ratio"]
                    )

                elif model_name == "random_forest":

                    mlflow.log_params(
                        self.config.models[
                            "random_forest"
                        ]
                    )

                elif model_name == "gradient_boosting":

                    mlflow.log_params(
                        self.config.models[
                            "gradient_boosting"
                        ]
                    )

                # ----------------------------------------------------
                # Log CV configuration
                # ----------------------------------------------------

                mlflow.log_param(
                    "cv_folds",
                    self.config.cv_folds
                )

                mlflow.log_param(
                    "cv_scoring",
                    self.config.cv_scoring
                )

                # ----------------------------------------------------
                # Cross Validation
                # ----------------------------------------------------

                cv_scores = cross_val_score(
                    model,
                    train_x,
                    train_y,
                    cv=self.config.cv_folds,
                    scoring=self.config.cv_scoring,
                    n_jobs=-1
                )

                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                logger.info(
                    f"{model_name} CV R2: "
                    f"{cv_mean:.4f} ± {cv_std:.4f}"
                )

                # MLflow CV metrics

                mlflow.log_metric(
                    "cv_r2_mean",
                    cv_mean
                )

                mlflow.log_metric(
                    "cv_r2_std",
                    cv_std
                )

                # ----------------------------------------------------
                # Train model
                # ----------------------------------------------------

                model.fit(
                    train_x,
                    train_y
                )

                # ----------------------------------------------------
                # Predictions
                # ----------------------------------------------------

                predictions = model.predict(
                    test_x
                )

                # ----------------------------------------------------
                # Metrics
                # ----------------------------------------------------

                rmse = np.sqrt(
                    mean_squared_error(
                        test_y,
                        predictions
                    )
                )

                mae = mean_absolute_error(
                    test_y,
                    predictions
                )

                r2 = r2_score(
                    test_y,
                    predictions
                )

                logger.info(
                    f"{model_name} -> "
                    f"RMSE: {rmse:.4f}, "
                    f"MAE: {mae:.4f}, "
                    f"R2: {r2:.4f}"
                )

                # ----------------------------------------------------
                # MLflow metrics
                # ----------------------------------------------------

                mlflow.log_metric(
                    "rmse",
                    rmse
                )

                mlflow.log_metric(
                    "mae",
                    mae
                )

                mlflow.log_metric(
                    "r2",
                    r2
                )

                # ----------------------------------------------------
                # Log trained model
                # ----------------------------------------------------

                mlflow.sklearn.log_model(
                    model,
                    "model"
                )

                # ----------------------------------------------------
                # Store results
                # ----------------------------------------------------

                results[model_name] = {

                    "model": model,

                    "cv_mean": cv_mean,

                    "cv_std": cv_std,

                    "rmse": rmse,

                    "mae": mae,

                    "r2": r2
                }

        # ============================================================
        # 6. SELECT BEST MODEL BASED ON CROSS-VALIDATION
        # ============================================================

        best_model_name = max(
            results,
            key=lambda name:
            results[name]["cv_mean"]
        )

        best_model = results[
            best_model_name
        ]["model"]

        best_cv_score = results[
            best_model_name
        ]["cv_mean"]

        logger.info(
            f"Best model based on CV: "
            f"{best_model_name}"
        )

        logger.info(
            f"Best CV R2: "
            f"{best_cv_score:.4f}"
        )

        # ============================================================
        # 7. HYPERPARAMETER TUNING
        # ============================================================

        if best_model_name == "random_forest":

            logger.info(
                "Starting Random Forest "
                "hyperparameter tuning..."
            )

            param_distributions = {

                "n_estimators": [
                    100,
                    200,
                    300,
                    400,
                    500
                ],

                "max_depth": [
                    None,
                    5,
                    10,
                    15,
                    20,
                    25
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "max_features": [
                    "sqrt",
                    "log2",
                    1.0
                ]
            }

            random_search = RandomizedSearchCV(

                estimator=RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                ),

                param_distributions=param_distributions,

                n_iter=20,

                cv=self.config.cv_folds,

                scoring=self.config.cv_scoring,

                random_state=42,

                n_jobs=-1,

                verbose=1
            )

            random_search.fit(
                train_x,
                train_y
            )

            best_model = (
                random_search.best_estimator_
            )

            logger.info(
                f"Best Random Forest parameters: "
                f"{random_search.best_params_}"
            )

            logger.info(
                f"Best tuned CV R2: "
                f"{random_search.best_score_:.4f}"
            )

            # ========================================================
            # MLflow run for tuned model
            # ========================================================

            with mlflow.start_run(
                run_name="random_forest_tuned"
            ):

                # ----------------------------------------------------
                # Model information
                # ----------------------------------------------------


                mlflow.set_tag(
                    "model_stage",
                    "production_candidate"
                )

                mlflow.set_tag(
                    "selection_reason",
                    "highest_cross_validation_r2"
                )

                mlflow.log_param(
                    "model_name",
                    "random_forest_tuned"
                )

                mlflow.log_param(
                    "cv_folds",
                    self.config.cv_folds
                )

                mlflow.log_param(
                    "cv_scoring",
                    self.config.cv_scoring
                )

                mlflow.log_param(
                    "search_type",
                    "RandomizedSearchCV"
                )

                mlflow.log_param(
                    "n_iter",
                    20
                )

                # ----------------------------------------------------
                # Best hyperparameters
                # ----------------------------------------------------

                for param_name, param_value in (
                    random_search.best_params_.items()
                ):

                    mlflow.log_param(
                        f"best_{param_name}",
                        param_value
                    )

                # ----------------------------------------------------
                # Tuned CV score
                # ----------------------------------------------------

                mlflow.log_metric(
                    "tuned_cv_r2",
                    random_search.best_score_
                )

                # ----------------------------------------------------
                # Final predictions
                # ----------------------------------------------------

                final_predictions = (
                    best_model.predict(test_x)
                )

                final_rmse = np.sqrt(
                    mean_squared_error(
                        test_y,
                        final_predictions
                    )
                )

                final_mae = mean_absolute_error(
                    test_y,
                    final_predictions
                )

                final_r2 = r2_score(
                    test_y,
                    final_predictions
                )

                # ----------------------------------------------------
                # Log final metrics
                # ----------------------------------------------------

                mlflow.log_metric(
                    "test_rmse",
                    final_rmse
                )

                mlflow.log_metric(
                    "test_mae",
                    final_mae
                )

                mlflow.log_metric(
                    "test_r2",
                    final_r2
                )

                # ----------------------------------------------------
                # Log tuned model
                # ----------------------------------------------------

                mlflow.sklearn.log_model(
                    best_model,
                    "model"
                )

        # ============================================================
        # 8. FINAL MODEL EVALUATION
        # ============================================================

        final_predictions = best_model.predict(
            test_x
        )

        final_rmse = np.sqrt(
            mean_squared_error(
                test_y,
                final_predictions
            )
        )

        final_mae = mean_absolute_error(
            test_y,
            final_predictions
        )

        final_r2 = r2_score(
            test_y,
            final_predictions
        )

        logger.info(
            "Final model performance:"
        )

        logger.info(
            f"RMSE: {final_rmse:.4f}"
        )

        logger.info(
            f"MAE: {final_mae:.4f}"
        )

        logger.info(
            f"R2: {final_r2:.4f}"
        )

        # ============================================================
        # 9. SAVE FINAL MODEL
        # ============================================================

        model_path = os.path.join(
            self.config.root_dir,
            self.config.model_name
        )

        joblib.dump(
            best_model,
            model_path
        )

        logger.info(
            f"Best model saved at: {model_path}"
        )

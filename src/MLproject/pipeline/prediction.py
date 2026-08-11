import joblib
import pandas as pd
from pathlib import Path


class PredictionPipeline:

    def __init__(self):
        self.model = joblib.load(
            Path("artifacts/model_trainer/model.joblib")
        )

    def predict(self, data):

        columns = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]

        data = pd.DataFrame(data, columns=columns)

        prediction = self.model.predict(data)

        return prediction
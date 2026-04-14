from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd

from MLproject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)

        data = request.form

        input_data = [
            float(data.get('fixed_acidity', 0)),
            float(data.get('volatile_acidity', 0)),
            float(data.get('citric_acid', 0)),
            float(data.get('residual_sugar', 0)),
            float(data.get('chlorides', 0)),
            float(data.get('free_sulfur_dioxide', 0)),
            float(data.get('total_sulfur_dioxide', 0)),
            float(data.get('density', 0)),
            float(data.get('pH', 0)),   
            float(data.get('sulphates', 0)),
            float(data.get('alcohol', 0))
        ]

        data_array = np.array(input_data).reshape(1, 11)

        obj = PredictionPipeline()
        predict = obj.predict(data_array)

        return render_template('results.html', prediction=round(predict[0], 2))

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
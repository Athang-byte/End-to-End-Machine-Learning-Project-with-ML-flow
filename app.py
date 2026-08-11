
from flask import Flask, render_template, request
import os


from MLproject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy"}, 200

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print(request.form)

        data = request.form

        input_data = [[
            float(data.get("fixed_acidity", 0)),
            float(data.get("volatile_acidity", 0)),
            float(data.get("citric_acid", 0)),
            float(data.get("residual_sugar", 0)),
            float(data.get("chlorides", 0)),
            float(data.get("free_sulfur_dioxide", 0)),
            float(data.get("total_sulfur_dioxide", 0)),
            float(data.get("density", 0)),
            float(data.get("pH", 0)),
            float(data.get("sulphates", 0)),
            float(data.get("alcohol", 0))
        ]]

        obj = PredictionPipeline()

        prediction = obj.predict(input_data)

        return render_template(
            "results.html",
            prediction=round(float(prediction[0]), 2)
        )

    except Exception as e:

        return str(e)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
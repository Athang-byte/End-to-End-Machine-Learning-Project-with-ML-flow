from app import app


def test_homepage():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_health():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_prediction():
    client = app.test_client()

    data = {
        "fixed_acidity": "7.4",
        "volatile_acidity": "0.70",
        "citric_acid": "0.00",
        "residual_sugar": "1.9",
        "chlorides": "0.076",
        "free_sulfur_dioxide": "11",
        "total_sulfur_dioxide": "34",
        "density": "0.9978",
        "pH": "3.51",
        "sulphates": "0.56",
        "alcohol": "9.4"
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200
# app.py
import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# load model safely with joblib
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # input from user
    features = np.array([list(data.values())])  # convert dict to array
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# your full code goes here
import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

@app.route("/")
def home():
    return "Heart Disease Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([list(data.values())])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

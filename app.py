from flask import Flask, request, jsonify, render_template_string
import joblib

# Load model
model = joblib.load("heart_disease_model.joblib")  # use joblib instead of pickle
app = Flask(__name__)

# Simple HTML template
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
</head>
<body style="font-family: Arial; text-align: center; margin-top: 50px;">
    <h2>Heart Disease Prediction</h2>
    <form action="/predict" method="post">
        Age: <input type="number" name="age" required><br><br>
        Sex (1=Male, 0=Female): <input type="number" name="sex" required><br><br>
        Chest Pain Type (cp): <input type="number" name="cp" required><br><br>
        Resting BP: <input type="number" name="trestbps" required><br><br>
        Cholesterol: <input type="number" name="chol" required><br><br>
        Fasting Blood Sugar > 120 (1/0): <input type="number" name="fbs" required><br><br>
        Resting ECG: <input type="number" name="restecg" required><br><br>
        Max Heart Rate: <input type="number" name="thalach" required><br><br>
        Exercise Induced Angina (1/0): <input type="number" name="exang" required><br><br>
        Oldpeak: <input type="text" name="oldpeak" required><br><br>
        Slope: <input type="number" name="slope" required><br><br>
        CA: <input type="number" name="ca" required><br><br>
        Thal: <input type="number" name="thal" required><br><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is defined %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(form_html)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Handle both form and JSON input
        if request.is_json:
            data = request.get_json()
        else:
            data = {k: float(v) for k, v in request.form.items()}
        
        features = [
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]
        
        prediction = model.predict([features])[0]
        result = "Heart Disease Detected 💔" if prediction == 1 else "No Heart Disease 😊"

        # If request is JSON → return API response
        if request.is_json:
            return jsonify({"prediction": int(prediction), "message": result})
        else:
            return render_template_string(form_html, prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

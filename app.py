from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled = scaler.transform([data])
    prediction = model.predict(scaled)[0]
    result = "High Risk (Will Die)" if prediction == 1 else "Low Risk (Will Survive)"
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True )
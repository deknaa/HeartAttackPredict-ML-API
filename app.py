from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/heart-attack-model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # prediksi menggunakan model yang sudah dilatih
    prediction = model.predict(df)
    result = "Risk" if prediction[0] == 1 else "No Risk"

    return jsonify({"Prediction": result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
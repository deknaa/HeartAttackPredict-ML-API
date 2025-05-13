from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# load model
model = joblib.load("models/heart-attack-model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari request
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])

    # Lakukan prediksi
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1] # array([[0.23, 0.77]])

    # Kembalikan hasil prediksi
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"Message": "Test GET API Success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
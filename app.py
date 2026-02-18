"""
Water Potability Prediction - Flask Web Application
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

FEATURE_INFO = {
    'ph': {'label': 'pH Level', 'min': 0, 'max': 14, 'step': 0.01,
           'default': 7.0, 'who': '6.5 – 8.5'},
    'Hardness': {'label': 'Hardness (mg/L)', 'min': 0, 'max': 400, 'step': 0.1,
                 'default': 196.0, 'who': '< 300'},
    'Solids': {'label': 'Total Dissolved Solids (ppm)', 'min': 0, 'max': 65000, 'step': 1,
               'default': 22000.0, 'who': '< 500'},
    'Chloramines': {'label': 'Chloramines (ppm)', 'min': 0, 'max': 15, 'step': 0.01,
                    'default': 7.1, 'who': '< 4'},
    'Sulfate': {'label': 'Sulfate (mg/L)', 'min': 0, 'max': 500, 'step': 0.1,
                'default': 333.0, 'who': '< 250'},
    'Conductivity': {'label': 'Conductivity (μS/cm)', 'min': 0, 'max': 800, 'step': 0.1,
                     'default': 426.0, 'who': '< 400'},
    'Organic_carbon': {'label': 'Organic Carbon (ppm)', 'min': 0, 'max': 30, 'step': 0.01,
                       'default': 14.0, 'who': '< 2'},
    'Trihalomethanes': {'label': 'Trihalomethanes (μg/L)', 'min': 0, 'max': 130, 'step': 0.01,
                        'default': 66.0, 'who': '< 80'},
    'Turbidity': {'label': 'Turbidity (NTU)', 'min': 0, 'max': 7, 'step': 0.01,
                  'default': 3.97, 'who': '< 5'}
}


@app.route('/')
def index():
    return render_template('index.html', features=FEATURE_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data.get(f, FEATURE_INFO[f]['default'])) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prediction = int(model.predict(arr_scaled)[0])
        probability = float(model.predict_proba(arr_scaled)[0][1])
        return jsonify({
            'prediction': prediction,
            'potable': bool(prediction == 1),
            'probability': round(probability * 100, 2),
            'label': 'POTABLE ✅' if prediction == 1 else 'NOT POTABLE ❌',
            'message': 'Safe for drinking' if prediction == 1 else 'Not safe for drinking'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'Gradient Boosting (Tuned)'})


if __name__ == '__main__':
    print("Starting Water Potability Prediction API...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

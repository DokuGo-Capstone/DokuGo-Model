from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)

# Muat model dan scaler
model = load_model('/home/rapuri/Python/capstone/expense_prediction_model_fixed.h5')  # Ganti dengan path ke model Anda
# scaler = joblib.load('/home/rapuri/Python/capstone/scaler.pkl')  # Muat scaler yang telah di-fit sebelumnya

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari request JSON
        data = request.json
        amount = data.get('amount', 0)
        amount /= 187.85
        lag_1_expenses = data.get('Lag_1_Expenses', 0)
        lag_2_expenses = data.get('Lag_2_Expenses', 0)
        category_encoded = data.get('category_encoded', 0)
        day_of_week = data.get('day_of_week', 0)
        is_weekend = data.get('is_weekend', 0)

        # Preprocessing input
        amount_log = np.log1p(amount)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        smoothed_expenses = amount_log
        rolling_avg_7 = amount_log
        rolling_avg_30 = amount_log

        # Gabungkan semua fitur
        input_features = [
            lag_1_expenses, lag_2_expenses, category_encoded,
            day_of_week_sin, day_of_week_cos, is_weekend,
            smoothed_expenses, rolling_avg_7, rolling_avg_30
        ]

        # Normalisasi input
        # input_scaled = scaler.transform([input_features])

        # Reshape untuk model (3D input)
        input_reshaped = np.array(input_features).reshape((1, len(input_features), 1))

        # Prediksi menggunakan model
        y_pred_log = model.predict(input_reshaped)[0][0]

        # Konversi hasil prediksi ke skala asli
        y_pred_original = np.expm1(y_pred_log)
        
        y_pred_original *= 187.85

        # Response
        response = {
            "Input Features (Preprocessed)": [float(f) for f in input_features],
            # "Input Features (Scaled)": [float(f) for f in input_scaled.flatten()],
            "Prediksi (Log Scale)": round(float(y_pred_log), 4),
            "Prediksi (Original Scale)": f"Rp. {y_pred_original:,.2f}"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

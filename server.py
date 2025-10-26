from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Carica dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
df = pd.read_csv(DATASET_PATH)
avg_student = df["Student stress level"].mean()

# (Opzionale) Carica modello se esiste
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "decision_tree_model.pkl")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Funzione di previsione (usa modello se disponibile)
def predict_stress(inputs):
    if model:
        df_inputs = pd.DataFrame([inputs],
            columns=["Sleep quality","Weeks headaches","Academic performance","Study load","Extracurricular activities weekly"])
        pred = model.predict(df_inputs)[0]
    else:
        # fallback con formula empirica
        weights = np.array([0.25, 0.2, -0.3, 0.4, -0.15])
        base = 3
        pred = base + np.dot(weights, np.array(inputs) - 3)
    return float(np.clip(pred, 1, 5))

# Endpoint /predict
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    answers = data.get("answers", [])
    if len(answers) != 5:
        return jsonify({"error": "Expected 5 answers"}), 400

    user_score = predict_stress(answers)
    return jsonify({
        "avgStudent": round(avg_student, 2),
        "userScore": round(user_score, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)

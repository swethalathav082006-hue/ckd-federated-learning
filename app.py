from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

FEATURE_ORDER = [
    "age", "al", "appet", "ba", "bgr", "bp", "bu", "hemo",
    "pc", "pcc", "pcv", "pot", "rbc", "rbcc", "sc", "sg",
    "sod", "su", "wbcc"
]

class CKDModel(nn.Module):
    def __init__(self, input_features=19):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)

model = CKDModel(input_features=len(FEATURE_ORDER)).to(DEVICE)
model_path = BASE_DIR / "models" / "adaptive_dp_fl_model.pth"

model_loaded = False
model_message = "Model file not found."

if model_path.exists():
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        model_loaded = True
        model_message = "Adaptive-DP model loaded."
    except Exception as exc:
        model_message = f"Could not load model: {exc}"

# Common text encodings for categorical inputs.
# These mappings must be checked against the encodings used during training.
CATEGORY_MAPS = {
    "appet": {"good": 1, "poor": 0},
    "ba": {"present": 1, "notpresent": 0, "no": 0, "yes": 1},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0, "no": 0, "yes": 1},
    "rbc": {"normal": 1, "abnormal": 0},
}

YES_NO_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0}

def parse_value(feature, value):
    if value is None or str(value).strip() == "":
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower()

    if feature in CATEGORY_MAPS and text in CATEGORY_MAPS[feature]:
        return float(CATEGORY_MAPS[feature][text])

    if text in YES_NO_MAP:
        return float(YES_NO_MAP[text])

    # Supports decimal numbers and values such as "120/80"
    try:
        return float(text)
    except ValueError:
        import re
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group(0))
        raise ValueError(
            f"Cannot interpret '{value}' for feature '{feature}'. "
            "Use a number or a supported category."
        )

def load_preprocessor(hospital_number):
    folder = BASE_DIR / "preprocessors"
    imputer_path = folder / f"hospital_{hospital_number}_imputer.joblib"
    scaler_path = folder / f"hospital_{hospital_number}_scaler.joblib"

    if not imputer_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Preprocessing files for Hospital {hospital_number} were not found."
        )

    return joblib.load(imputer_path), joblib.load(scaler_path)

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "model_message": model_message,
        "mode": "research demonstration only",
        "feature_count": len(FEATURE_ORDER)
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({
            "error": "Adaptive-DP model is unavailable.",
            "details": model_message
        }), 503

    payload = request.get_json(silent=True) or {}
    hospital = payload.get("hospital", 1)
    features = payload.get("features", payload.get("data", {}))

    try:
        hospital = int(hospital)
        if hospital not in [1, 2, 3, 4]:
            raise ValueError("Hospital must be 1, 2, 3, or 4.")

        if isinstance(features, list):
            if len(features) != len(FEATURE_ORDER):
                raise ValueError("Provide exactly 19 feature values.")
            row = dict(zip(FEATURE_ORDER, features))
        elif isinstance(features, dict):
            row = features
        else:
            raise ValueError("'features' must be an object or a list.")

        raw = {
            feature: parse_value(feature, row.get(feature))
            for feature in FEATURE_ORDER
        }

        X = pd.DataFrame([raw], columns=FEATURE_ORDER)

        imputer, scaler = load_preprocessor(hospital)
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            probability = torch.sigmoid(model(tensor)).item()

        # Exploratory display threshold only; not a clinical threshold.
        threshold = 0.60
        predicted_class = int(probability >= threshold)

        return jsonify({
            "prediction": (
                "CKD risk signal" if predicted_class == 1
                else "No CKD risk signal at the selected threshold"
            ),
            "predicted_class": predicted_class,
            "probability": round(float(probability), 4),
            "threshold": threshold,
            "hospital": hospital,
            "features_used": FEATURE_ORDER,
            "warning": (
                "Research demonstration only. Not a medical diagnosis. "
                "Preprocessing compatibility with the original training "
                "pipeline must be verified before interpreting predictions."
            )
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

import json
import joblib
import pandas as pd
import os

MODEL_DIR = "/opt/ml/model"


def model_fn(model_dir=MODEL_DIR):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model_objects):
    model, scaler = model_objects
    X_scaled = scaler.transform(input_data)
    return model.predict(X_scaled)


def output_fn(predictions, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(predictions.tolist())
    raise ValueError("Unsupported accept type")

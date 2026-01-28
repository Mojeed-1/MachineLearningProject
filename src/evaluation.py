import json
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocessing import preprocess_data


def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_dir: str = "evaluation",
):
    """
    Evaluate Linear Regression model and save regression metrics.
    """

    model = joblib.load(model_path)

    df = pd.read_csv(test_data_path)
    X_test, y_test = preprocess_data(df, label_column="label")

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=4))

    return metrics

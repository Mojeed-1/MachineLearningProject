import os
import pandas as pd
from src.model import get_model
from src.data_loader import load_data
import joblib


def main():
    train_path = os.getenv("SM_CHANNEL_TRAIN", "data/train/BostonHousing.csv")
    model_dir = os.getenv("SM_MODEL_DIR", "model/")

    df = load_data(train_path)
    X = df.drop("label", axis=1)
    y = df["label"]

    model = get_model()
    model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    main()

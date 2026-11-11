"""Load the saved model and predict a crop from seven measurements."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "knn_model.pkl"


def predict_crop(values, model_path=MODEL_PATH):
    """Return the predicted label; no confidence score is claimed or displayed."""
    if len(values) != len(FEATURES):
        raise ValueError(f"Provide exactly {len(FEATURES)} feature values.")
    row = np.asarray(values, dtype=float)
    if not np.isfinite(row).all():
        raise ValueError("All input values must be finite numbers.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Add the dataset and run: python -m src.train_model")
    model = joblib.load(model_path)
    return str(model.predict(pd.DataFrame([row], columns=FEATURES))[0])

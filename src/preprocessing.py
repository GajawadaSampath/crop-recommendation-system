"""Dataset loading and validation for the crop recommendation dataset."""
from pathlib import Path

import pandas as pd
import numpy as np

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
REQUIRED_COLUMNS = FEATURES + [TARGET]
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "dataset" / "Crop_recommendation.csv"


def load_dataset(path=DEFAULT_DATASET):
    """Load the supplied CSV and report missing/duplicate data before training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download Crop_recommendation.csv from "
            "the Kaggle source in README.md and place it in the dataset/ folder."
        )
    data = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    data = data[REQUIRED_COLUMNS].copy()
    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())
    if missing_count:
        raise ValueError(f"Dataset contains {missing_count} missing cell(s); clean the CSV first.")
    if duplicate_count:
        print(f"Found {duplicate_count} duplicate row(s); dropping duplicates for training.")
        data = data.drop_duplicates().reset_index(drop=True)
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[FEATURES].isna().any().any():
        raise ValueError("One or more feature columns contain non-numeric values.")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Feature values must all be finite numbers.")
    if data[TARGET].astype(str).str.strip().eq("").any():
        raise ValueError("Target labels must not be empty.")
    data[TARGET] = data[TARGET].astype(str).str.strip()
    if data[TARGET].nunique() < 2:
        raise ValueError("At least two crop classes are needed for classification.")
    print(f"Loaded {len(data)} rows and {data[TARGET].nunique()} crop classes.")
    print("Class distribution:\n" + data[TARGET].value_counts().sort_index().to_string())
    return data

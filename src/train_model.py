"""Select K using training-only cross-validation and save the fitted KNN pipeline."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import DEFAULT_DATASET, FEATURES, TARGET, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "knn_model.pkl"
RESULTS_PATH = ROOT / "reports" / "model_results.json"
K_VALUES = [3, 5, 7, 9, 11]
RANDOM_STATE = 42


def train(dataset_path=DEFAULT_DATASET):
    data = load_dataset(dataset_path)
    X, y = data[FEATURES], data[TARGET]
    # Keep a stratified 20% test partition untouched during model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    if y_train.value_counts().min() < 5:
        raise ValueError("Each class needs at least five training rows for 5-fold stratified CV.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for k in K_VALUES:
        # Scaling is fitted separately inside each CV fold to prevent leakage.
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))])
        scores[str(k)] = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy").tolist()
    mean_scores = {k: float(np.mean(values)) for k, values in scores.items()}
    best_k = int(max(mean_scores, key=mean_scores.get))
    # Fit preprocessing and model on the full training partition only.
    final_model = Pipeline([("scaler", StandardScaler()),
                            ("knn", KNeighborsClassifier(n_neighbors=best_k, metric="euclidean"))])
    final_model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    results = {
        "status": "trained_pending_evaluation", "dataset": str(Path(dataset_path).name),
        "records_after_deduplication": len(data), "class_count": int(y.nunique()),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "features": FEATURES, "target": TARGET, "test_size": 0.2,
        "random_state": RANDOM_STATE, "cross_validation_folds": 5,
        "candidate_k": K_VALUES, "cv_accuracy_by_k": mean_scores,
        "cv_fold_accuracy_by_k": scores, "selected_k": best_k,
        "train_records": len(X_train), "test_records": len(X_test),
        "training_accuracy": float(final_model.score(X_train, y_train)),
        "model_path": "models/knn_model.pkl"
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected K={best_k}; saved model to {MODEL_PATH}")
    return final_model


if __name__ == "__main__":
    train()

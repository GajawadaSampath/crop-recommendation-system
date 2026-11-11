"""Evaluate the saved pipeline against a reproducible held-out test partition."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from src.preprocessing import DEFAULT_DATASET, FEATURES, TARGET, load_dataset
from src.train_model import MODEL_PATH, RESULTS_PATH, RANDOM_STATE

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def evaluate(dataset_path=DEFAULT_DATASET):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python -m src.train_model first.")
    data = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, stratify=data[TARGET], random_state=RANDOM_STATE
    )
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results.update({
        "status": "evaluated", "training_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "testing_accuracy": float(accuracy_score(y_test, predicted)), "weighted_precision": float(precision),
        "weighted_recall": float(recall), "weighted_f1_score": float(f1),
        "classification_report": report,
        "confusion_matrix_labels": sorted(data[TARGET].unique().tolist()),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=sorted(data[TARGET].unique())).tolist()
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    labels = results["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(results["confusion_matrix"], xticklabels=labels, yticklabels=labels,
                cmap="Greens", annot=True, fmt="d", ax=ax)
    ax.set(title="KNN Confusion Matrix (Held-out Test Set)", xlabel="Predicted crop", ylabel="Actual crop")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrix.png", dpi=180); plt.close(fig)
    cv = results["cv_accuracy_by_k"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(k) for k in cv], list(cv.values()), marker="o", color="#26834a")
    ax.set(title="K Value vs 5-fold CV Accuracy", xlabel="Number of neighbors (K)", ylabel="Mean CV accuracy")
    ax.set_xticks([int(k) for k in cv]); ax.set_ylim(0, 1.02); fig.tight_layout()
    fig.savefig(FIGURES / "k_value_comparison.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    data[TARGET].value_counts().sort_index().plot(kind="bar", color="#5ba56c", ax=ax)
    ax.set(title="Crop Class Distribution", xlabel="Crop", ylabel="Records"); fig.tight_layout()
    fig.savefig(FIGURES / "crop_distribution.png", dpi=180); plt.close(fig)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Keep the academic report synchronized with the genuine evaluation outputs.
    from src.generate_report import generate_report
    generate_report()
    print(f"Held-out accuracy: {results['testing_accuracy']:.4f}; results and figures saved.")
    return results


if __name__ == "__main__":
    evaluate()

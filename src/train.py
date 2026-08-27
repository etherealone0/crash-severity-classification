"""Baseline, SMOTENC-balanced, and final model training/evaluation."""

import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RANDOM_STATE = 42
LABELS = ["Minor", "Moderate", "Severe"]

# SVC(kernel="rbf") scales roughly O(n^2-n^3) with training set size, which makes
# training directly on the full ~318K-row training split impractical (hours to
# days). We stratify-subsample the training set down to SVC_TRAIN_SIZE rows for
# Tasks 6-8; the full test set is still used for evaluation.
SVC_TRAIN_SIZE = 30_000


def load_train_test():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    return train, test


def subsample_for_svc(train, n=SVC_TRAIN_SIZE, random_state=RANDOM_STATE):
    """Stratified subsample of the training split, sized for tractable SVC training."""
    frac = min(n / len(train), 1.0)
    sub = train.groupby("severity_class", group_keys=False).sample(frac=frac, random_state=random_state)
    return sub.sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_xy(df):
    X = df.drop(columns=["severity_class"])
    y = df["severity_class"]
    return X, y


def evaluate(model, X_test, y_test, labels=LABELS):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0),
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


def print_metrics(metrics):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    for label in LABELS:
        pc = metrics["per_class"][label]
        print(f"  {label}: precision={pc['precision']:.3f} recall={pc['recall']:.3f} f1={pc['f1-score']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):", metrics["confusion_matrix"]["labels"])
    for row in metrics["confusion_matrix"]["matrix"]:
        print(row)


def write_metrics(metrics, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out_path}")


def train_baseline():
    """Task 6: SVC(kernel='rbf'), default hyperparameters, no class balancing."""
    train, test = load_train_test()
    train_sub = subsample_for_svc(train)
    print(f"Training SVC on {len(train_sub)} rows (subsampled from {len(train)} for tractability)")
    print(f"Evaluating on {len(test)} test rows")

    X_train, y_train = split_xy(train_sub)
    X_test, y_test = split_xy(test)

    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["train_rows"] = len(train_sub)
    metrics["train_rows_full_split"] = len(train)
    metrics["test_rows"] = len(test)
    metrics["note"] = (
        f"Trained on a {len(train_sub)}-row stratified subsample of the {len(train)}-row "
        "training split; SVC(kernel='rbf') does not scale to the full split in practical time."
    )

    print()
    print_metrics(metrics)
    write_metrics(metrics, "baseline_metrics.json")
    return model, metrics


if __name__ == "__main__":
    train_baseline()

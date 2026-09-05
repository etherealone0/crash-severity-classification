"""Baseline and SMOTENC-balanced SVC training/evaluation."""

import json
import os
import sys

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from data_prep import BOOLEAN_FEATURES  # noqa: E402

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RANDOM_STATE = 42
LABELS = ["Minor", "Moderate", "Severe"]

# SVC(kernel="rbf") scales roughly O(n^2-n^3) with training set size, which makes
# training directly on the full ~318K-row training split impractical (hours to
# days). We stratify-subsample the training set down to SVC_TRAIN_SIZE rows for
# Tasks 6-8; the full test set is still used for evaluation. 30K rows trains fast
# on the (trivially separable) imbalanced baseline, but once SMOTENC balances the
# classes the problem becomes genuinely hard to separate and libsvm's fit time
# blows up (15+ min and still running at 30K); 8K keeps every variant tractable,
# including the repeated fits Optuna needs in Task 8.
SVC_TRAIN_SIZE = 8_000


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


def categorical_feature_indices(X):
    """Column positions of the categorical features, for SMOTENC's categorical_features arg.
    Booleans (Amenity, Junction, ...), the Sunrise_Sunset_Night/Civil_Twilight_Night
    lighting flags, and the one-hot DayOfWeek_* columns are categorical; the weather
    numerics, Distance(mi), Hour, and the two frequency-encoded columns are continuous."""
    categorical_cols = (
        list(BOOLEAN_FEATURES)
        + [c for c in X.columns if c.endswith("_Night")]
        + [c for c in X.columns if c.startswith("DayOfWeek_")]
    )
    return [X.columns.get_loc(c) for c in categorical_cols]


def train_smotenc():
    """Task 7: SMOTENC-balance the same training subsample, retrain the same SVC,
    evaluate on the same untouched test set from Task 5/6."""
    train, test = load_train_test()
    train_sub = subsample_for_svc(train)
    X_train, y_train = split_xy(train_sub)
    X_test, y_test = split_xy(test)

    cat_idx = categorical_feature_indices(X_train)
    print(f"SMOTENC: {len(cat_idx)} categorical features out of {X_train.shape[1]}")
    print("Class counts before SMOTENC:", y_train.value_counts().to_dict())

    smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Class counts after SMOTENC:", y_res.value_counts().to_dict())

    model = SVC(kernel="rbf")
    model.fit(X_res, y_res)

    metrics = evaluate(model, X_test, y_test)
    metrics["train_rows_before_smote"] = len(train_sub)
    metrics["train_rows_after_smote"] = len(X_res)
    metrics["test_rows"] = len(test)
    metrics["note"] = (
        f"SMOTENC-balanced the {len(train_sub)}-row SVC training subsample up to "
        f"{len(X_res)} rows (equal class counts), then trained the same default-"
        "hyperparameter SVC(kernel='rbf') and evaluated on the same test set as the baseline."
    )

    print()
    print_metrics(metrics)
    write_metrics(metrics, "smote_metrics.json")
    return model, metrics


if __name__ == "__main__":
    train_baseline()
    print("\n" + "=" * 60 + "\n")
    train_smotenc()

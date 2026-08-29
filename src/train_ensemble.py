"""LightGBM tree-ensemble variant: escapes SVC's scaling limit by training on the
full training split, and uses class_weight="balanced" as an alternative to SMOTENC
for handling the class imbalance."""

import os
import sys

import joblib
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(__file__))
from train import RANDOM_STATE, evaluate, load_train_test, print_metrics, split_xy, write_metrics  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_lgbm.joblib")


def train_lgbm():
    """LGBMClassifier(class_weight='balanced'), trained on the full training split --
    unlike SVC(kernel='rbf'), gradient-boosted trees don't have an O(n^2-n^3) scaling
    problem, so no subsampling is needed."""
    train, test = load_train_test()
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    print(f"Training LightGBM on {len(X_train)} rows (full training split, no subsampling)")
    print(f"Evaluating on {len(test)} test rows")

    model = LGBMClassifier(objective="multiclass", class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(test)
    metrics["note"] = (
        f"LGBMClassifier(class_weight='balanced'), trained on the full {len(X_train)}-row "
        "training split (vs. the 8,000-row subsample SVC needs) since gradient-boosted "
        "trees don't share SVC's scaling limit."
    )

    print()
    print_metrics(metrics)
    write_metrics(metrics, "lgbm_metrics.json")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nWrote {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_lgbm()

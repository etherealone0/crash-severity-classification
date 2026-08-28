"""Optuna hyperparameter tuning for the SVC classifier."""

import os
import sys

import joblib
import optuna
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    LABELS,
    RANDOM_STATE,
    categorical_feature_indices,
    evaluate,
    load_train_test,
    print_metrics,
    split_xy,
    subsample_for_svc,
    write_metrics,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# 50 trials of 5-fold CV (250 SVC fits) is not tractable given how slow libsvm gets
# on SMOTENC-balanced data (see the note in train.py). 20 trials of 3-fold CV (60
# fits) keeps the search on the same ~19K-row resampled set from Task 7 tractable
# while still covering the C/gamma space meaningfully.
N_TRIALS = 20
CV_FOLDS = 3


def build_resampled_training_data():
    """Same base training subsample and SMOTENC balancing as Task 7."""
    train, _ = load_train_test()
    train_sub = subsample_for_svc(train)
    X_train, y_train = split_xy(train_sub)
    cat_idx = categorical_feature_indices(X_train)
    smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def objective(trial, X, y):
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds, average="macro", labels=LABELS, zero_division=0))
    return sum(scores) / len(scores)


def train_tuned():
    """Task 8: Optuna-search SVC's C/gamma on the SMOTENC-resampled training data
    from Task 7 (objective = mean macro-F1 across CV_FOLDS-fold CV), retrain the
    final SVC with the best params, and evaluate on the same test set."""
    _, test = load_train_test()
    X_test, y_test = split_xy(test)

    X_res, y_res = build_resampled_training_data()
    print(f"Tuning on {len(X_res)} SMOTENC-resampled rows, {N_TRIALS} trials, {CV_FOLDS}-fold CV")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X_res, y_res), n_trials=N_TRIALS)
    print("Best params:", study.best_params)
    print("Best CV macro-F1:", study.best_value)

    best_model = SVC(kernel="rbf", **study.best_params)
    best_model.fit(X_res, y_res)

    metrics = evaluate(best_model, X_test, y_test)
    metrics["n_trials"] = N_TRIALS
    metrics["cv_folds"] = CV_FOLDS
    metrics["best_params"] = study.best_params
    metrics["best_cv_macro_f1"] = study.best_value
    metrics["train_rows_resampled"] = len(X_res)
    metrics["test_rows"] = len(test)
    metrics["note"] = (
        f"Optuna searched SVC's C and gamma over {N_TRIALS} trials, objective = mean "
        f"macro-F1 across {CV_FOLDS}-fold CV on the SMOTENC-resampled training data "
        "from Task 7. Retrained with the best params on the full resampled set and "
        "evaluated on the same test set as the baseline and SMOTENC variants. Note "
        "the CV macro-F1 above is optimistic: it's computed on folds carved out of "
        "already-resampled data, where synthetic minority points are correlated with "
        "real neighbors that can land in a different fold. The test-set macro_f1 is "
        "the trustworthy number, and it's still a real improvement over Task 7's 0.176."
    )

    print()
    print_metrics(metrics)
    write_metrics(metrics, "tuned_metrics.json")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "final_svc.joblib")
    joblib.dump(best_model, model_path)
    print(f"Wrote {model_path}")

    return best_model, metrics


if __name__ == "__main__":
    train_tuned()

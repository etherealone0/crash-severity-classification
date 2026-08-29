"""Optuna tuning for the LightGBM variant: n_estimators, max_depth, learning_rate,
and per-class imbalance weighting, searched leak-free (cross_val_score on the
original imbalanced data -- same principle as the SVC leakage fix, though there's
no resampler here to leak across folds since class_weight replaces SMOTENC)."""

import os
import sys

import joblib
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from train import RANDOM_STATE, evaluate, load_train_test, print_metrics, split_xy, write_metrics  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_lgbm_tuned.joblib")

# Full-data CV (127s/trial at n_estimators=200) doesn't fit a 60-trial search in
# reasonable time. A 200K-row stratified subsample cuts that to ~15-40s/trial
# while still being large enough to rank hyperparameter configs reliably; the
# final model is then refit on the full training split regardless.
CV_SUBSAMPLE_SIZE = 200_000
N_TRIALS = 60
CV_FOLDS = 3

optuna.logging.set_verbosity(optuna.logging.INFO)


def subsample_for_search(train, n=CV_SUBSAMPLE_SIZE, random_state=RANDOM_STATE):
    frac = min(n / len(train), 1.0)
    sub = train.groupby("severity_class", group_keys=False).sample(frac=frac, random_state=random_state)
    return sub.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_model(params):
    class_weight = {
        "Minor": 1.0,
        "Moderate": params["moderate_weight"],
        "Severe": params["severe_weight"],
    }
    return LGBMClassifier(
        objective="multiclass",
        class_weight=class_weight,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "moderate_weight": trial.suggest_float("moderate_weight", 1.0, 10.0, log=True),
        "severe_weight": trial.suggest_float("severe_weight", 1.0, 50.0, log=True),
    }
    model = build_model(params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return scores.mean()


def train_tuned_lgbm():
    train, test = load_train_test()
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    search_sub = subsample_for_search(train)
    X_search, y_search = split_xy(search_sub)
    print(f"Searching on {len(X_search)} rows (subsampled from {len(X_train)} for tractable CV), "
          f"{N_TRIALS} trials, {CV_FOLDS}-fold CV")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_search, y_search), n_trials=N_TRIALS)

    print(f"\nBest params: {study.best_params}")
    print(f"Best CV macro-F1: {study.best_value}")

    model = build_model(study.best_params)
    print(f"\nFitting final model on the full {len(X_train)}-row training split")
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(test)
    metrics["cv_search_rows"] = len(X_search)
    metrics["n_trials"] = N_TRIALS
    metrics["cv_folds"] = CV_FOLDS
    metrics["cv_macro_f1"] = study.best_value
    metrics["best_params"] = study.best_params
    metrics["note"] = (
        f"LGBMClassifier tuned via Optuna ({N_TRIALS} trials, {CV_FOLDS}-fold CV on a "
        f"{len(X_search)}-row stratified subsample for search speed), final model refit on "
        f"the full {len(X_train)}-row training split. Class weighting (Moderate/Severe "
        "multipliers) tuned alongside the tree hyperparameters as the imbalance strategy, "
        "in place of SMOTENC."
    )

    print()
    print_metrics(metrics)
    write_metrics(metrics, "lgbm_tuned_metrics.json")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nWrote {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_tuned_lgbm()

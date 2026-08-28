"""Optuna hyperparameter tuning for the SVC classifier."""

import os
import sys
import warnings

import joblib
import optuna
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
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

# 50 trials of 5-fold CV is not tractable given how slow libsvm gets on SMOTENC-
# balanced data (see the note in train.py). Refitting SMOTENC fresh inside each
# fold (the Task 15 leakage fix) makes each trial 5-6x slower than the original
# leaky version (~4-13 min/trial vs ~1-2 min), since every fold now gets its own
# independently-generated synthetic points instead of reusing one global
# resampling -- a harder, more varied optimization landscape for libsvm. 8 trials
# of 3-fold CV keeps this tractable while still covering the C/gamma space.
N_TRIALS = 8
CV_FOLDS = 3

# Some (low-C, low-gamma) corners of the search space make libsvm's SMO solver
# converge extremely slowly on this data -- one uncapped trial ran 2+ hours
# before it finally finished, and scored a mediocre 0.106 anyway. Capping
# max_iter bounds worst-case wall-clock time per fit; a trial that hits the cap
# without fully converging just gets scored on wherever it landed, which costs
# nothing in practice since those slow corners score poorly regardless.
SVC_MAX_ITER = 20_000


def build_cv_pipeline(cat_idx, C, gamma):
    """SMOTENC + SVC in one imblearn Pipeline (not sklearn's plain Pipeline, which
    can't carry a resampler). Used with cross_val_score so SMOTENC is refit fresh
    inside each fold on that fold's training data only -- each validation fold
    stays real, imbalanced, untouched data. This is the fix for Task 8's original
    CV leakage, where SMOTENC was applied once to the whole training subsample
    *before* splitting into folds, so synthetic minority points generated from
    the eventual validation rows could leak into the training folds (and vice
    versa), inflating the CV score to ~0.85 against a true test score of 0.350."""
    return Pipeline(
        [
            ("smotenc", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
            ("svc", SVC(kernel="rbf", C=C, gamma=gamma, max_iter=SVC_MAX_ITER)),
        ]
    )


def objective(trial, X, y, cat_idx):
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

    pipeline = build_cv_pipeline(cat_idx, C, gamma)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro")
    return scores.mean()


def train_tuned():
    """Task 8 (leakage fixed in Task 15): Optuna-search SVC's C/gamma with the
    objective evaluated via an imblearn Pipeline + cross_val_score on the
    *original imbalanced* training subsample, so SMOTENC gets refit per fold
    instead of once before splitting. Retrain the final SVC with the best
    params on a single SMOTENC pass over the full subsample, and evaluate on
    the same untouched test set as the baseline and SMOTENC variants."""
    train, test = load_train_test()
    train_sub = subsample_for_svc(train)
    X_train, y_train = split_xy(train_sub)
    X_test, y_test = split_xy(test)

    cat_idx = categorical_feature_indices(X_train)
    print(
        f"Tuning on {len(X_train)} rows (pre-SMOTENC, class counts "
        f"{y_train.value_counts().to_dict()}), {N_TRIALS} trials, {CV_FOLDS}-fold CV "
        "with SMOTENC refit inside each fold"
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X_train, y_train, cat_idx), n_trials=N_TRIALS)
    print("Best params:", study.best_params)
    print("Best CV macro-F1:", study.best_value)

    # Final fit: one SMOTENC pass over the full training subsample (no CV here,
    # so no leakage risk), then train the final model with the best params.
    smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    best_model = SVC(kernel="rbf", max_iter=SVC_MAX_ITER, **study.best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        best_model.fit(X_res, y_res)

    metrics = evaluate(best_model, X_test, y_test)
    metrics["n_trials"] = N_TRIALS
    metrics["cv_folds"] = CV_FOLDS
    metrics["best_params"] = study.best_params
    metrics["best_cv_macro_f1"] = study.best_value
    metrics["train_rows_resampled"] = len(X_res)
    metrics["test_rows"] = len(test)
    metrics["svc_max_iter"] = SVC_MAX_ITER
    metrics["note"] = (
        f"Optuna searched SVC's C and gamma over {N_TRIALS} trials, objective = mean "
        f"macro-F1 across {CV_FOLDS}-fold CV using an imblearn Pipeline (SMOTENC + SVC) "
        "so SMOTENC is refit inside each fold on that fold's training data only, keeping "
        "validation folds real and imbalanced. This fixes the CV-score leakage from the "
        "original Task 8 approach (SMOTENC applied once before splitting into folds, "
        f"which inflated CV macro-F1 to ~0.85 against a true test score of 0.350). SVC's "
        f"max_iter is capped at {SVC_MAX_ITER} since some hyperparameter corners converge "
        "extremely slowly (one uncapped fit ran 2+ hours); a capped fit that hasn't fully "
        "converged just gets scored on wherever it landed. Retrained with the best params "
        "on the full SMOTENC-resampled training data and evaluated on the same test set "
        "as the baseline and SMOTENC variants."
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

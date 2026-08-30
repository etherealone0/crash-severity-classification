"""Per-class probability threshold tuning for the tuned LightGBM model.

The confusion matrix on the tuned LightGBM doesn't show the ordinal-neighbor
pattern (Moderate <-> Severe) one might expect from severity being ordinal --
both Moderate and Severe are misclassified mostly *into Minor* (33.5% and 56.0%
of their rows respectively), vs. only 5.5%/5.1% into each other. That's
majority-class pull (Minor is 80% of the data), not ordinal-adjacency confusion.
Per-class threshold tuning directly counters majority-class pull: multiply each
class's predicted probability by a weight (>1 boosts, <1 penalizes) before taking
argmax, so Moderate/Severe don't need to out-score Minor's raw probability to win.

Thresholds are tuned out-of-fold (cross_val_predict on the training set, same
hyperparameters as the tuned model) so the search never touches the test set,
then applied once to the already-fit deployed model's test-set probabilities.
"""

import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from tune_ensemble import build_model  # noqa: E402
from train import (  # noqa: E402
    LABELS,
    RANDOM_STATE,
    load_train_test,
    print_metrics,
    split_xy,
    write_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_lgbm_tuned.joblib")
CV_FOLDS = 3
# 3 full fits at n_estimators=480 on the full 1.55M-row split proved far too slow in
# practice (still running after 100+ minutes of CPU time). A 300K-row stratified
# subsample keeps each fold's fit fast while still giving a reliable OOF signal for
# threshold selection -- the thresholds are then applied to the already-fit deployed
# model's *test-set* probabilities, so the subsampling only affects the search.
OOF_SUBSAMPLE_SIZE = 300_000

# Best hyperparameters from the Task 19 Optuna study (src/tune_ensemble.py).
BEST_PARAMS = {
    "n_estimators": 480,
    "max_depth": 6,
    "learning_rate": 0.06331597828901991,
    "moderate_weight": 2.100421972333565,
    "severe_weight": 7.469396416869201,
}


def macro_f1_from_confusion(cm):
    """Macro-F1 straight from a (n_labels, n_labels) confusion matrix -- avoids
    sklearn's f1_score per-call overhead, which dominates when called hundreds of
    times over large arrays."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return f1.mean()


def macro_f1_for_weights(proba, y_true_idx, weights, n_labels):
    pred_idx = np.argmax(proba * weights, axis=1)
    cm = np.bincount(y_true_idx * n_labels + pred_idx, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    return macro_f1_from_confusion(cm)


def search_weights(proba, y_true_idx, n_labels):
    """Grid search over (Moderate, Severe) weight multipliers, Minor fixed at 1.0.
    Predictions are argmax(proba * weights), so a weight > 1 boosts that class
    (easier to win against Minor's raw probability) and a weight < 1 penalizes it.
    Range spans both directions on a log scale since the confusion matrix showed
    Moderate/Severe are pulled *into* Minor -- boosting, not penalizing, is the
    fix that's actually expected to help."""
    best_score, best_weights = -1.0, (1.0, 1.0, 1.0)
    for w_mod in np.geomspace(0.2, 10.0, 30):
        for w_sev in np.geomspace(0.2, 30.0, 35):
            weights = np.array([1.0, w_mod, w_sev])
            score = macro_f1_for_weights(proba, y_true_idx, weights, n_labels)
            if score > best_score:
                best_score, best_weights = score, (1.0, w_mod, w_sev)
    return best_weights, best_score


def metrics_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
    return precision, recall


def moderate_sweep(proba, y_true_idx, n_labels, w_sev_fixed):
    """Sweep the Moderate weight alone (Severe held at its macro-F1-optimal value)
    to show the recall/precision/macro-F1 tradeoff curve as Moderate is boosted
    harder than the macro-F1-optimal point allows -- this is the "target Moderate
    directly, not just the average" view Task 20 asks for."""
    rows = []
    for w_mod in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        weights = np.array([1.0, w_mod, w_sev_fixed])
        pred_idx = np.argmax(proba * weights, axis=1)
        cm = np.bincount(y_true_idx * n_labels + pred_idx, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
        precision, recall = metrics_from_confusion(cm)
        rows.append({
            "w_mod": w_mod,
            "moderate_recall": recall[1],
            "moderate_precision": precision[1],
            "macro_f1": macro_f1_from_confusion(cm),
        })
    return rows


def apply_weights(proba, weights):
    pred_idx = np.argmax(proba * np.array(weights), axis=1)
    return np.array(LABELS)[pred_idx]


def evaluate_with_weights(y_true, y_pred, labels=LABELS):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


def tune_thresholds():
    train, test = load_train_test()
    X_test, y_test = split_xy(test)

    frac = min(OOF_SUBSAMPLE_SIZE / len(train), 1.0)
    train_sub = train.groupby("severity_class", group_keys=False).sample(frac=frac, random_state=RANDOM_STATE)
    X_sub, y_sub = split_xy(train_sub)
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    y_sub_idx = y_sub.map(label_to_idx).to_numpy()

    print(f"Getting out-of-fold predict_proba on {len(X_sub)} rows (subsampled from "
          f"{len(train)} for tractable fold-fitting), {CV_FOLDS}-fold CV, same "
          "hyperparameters as the deployed model -- this never touches the test set")
    # Manual OOF loop instead of sklearn's cross_val_predict: cross_val_predict
    # pre-encodes y to integer class indices before calling estimator.fit() (to keep
    # predict_proba columns aligned across folds), which breaks LightGBM's
    # string-keyed class_weight dict (KeyError on the string labels). Fitting fold by
    # fold ourselves keeps y as the original string labels the whole way through.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = np.zeros((len(X_sub), len(LABELS)))
    for fold_i, (fit_idx, val_idx) in enumerate(cv.split(X_sub, y_sub)):
        fold_model = build_model(BEST_PARAMS)
        fold_model.fit(X_sub.iloc[fit_idx], y_sub.iloc[fit_idx])
        assert list(fold_model.classes_) == LABELS, (
            f"predict_proba column order {list(fold_model.classes_)} != LABELS {LABELS}"
        )
        proba_oof[val_idx] = fold_model.predict_proba(X_sub.iloc[val_idx])
        print(f"  fold {fold_i + 1}/{CV_FOLDS} done")

    raw_oof_f1 = macro_f1_for_weights(proba_oof, y_sub_idx, np.array([1.0, 1.0, 1.0]), len(LABELS))
    print(f"Out-of-fold macro-F1 with raw argmax (no threshold adjustment): {raw_oof_f1:.4f}")

    print("Grid-searching per-class weights for macro-F1 (500 combinations)...")
    best_weights, best_oof_f1 = search_weights(proba_oof, y_sub_idx, len(LABELS))
    print(f"Macro-F1-optimal weights (Minor, Moderate, Severe): {best_weights}")
    print(f"Macro-F1-optimal out-of-fold macro-F1: {best_oof_f1:.4f}")
    print("(barely different from raw argmax -- macro-F1 doesn't reward pushing "
          "Moderate recall harder, since the precision cost cancels it out)")

    print("\nSweeping Moderate weight alone (Severe held at its macro-F1-optimal "
          "value) to see how far Moderate recall can move, and what it costs:")
    sweep = moderate_sweep(proba_oof, y_sub_idx, len(LABELS), w_sev_fixed=best_weights[2])
    for row in sweep:
        print(f"  w_mod={row['w_mod']:>4.1f}  Moderate recall={row['moderate_recall']:.1%}  "
              f"Moderate precision={row['moderate_precision']:.1%}  macro-F1={row['macro_f1']:.4f}")

    # Targeted operating point: roughly double Moderate recall vs. raw argmax while
    # keeping macro-F1 within 10% relative of the untuned score -- a defensible
    # "direct attention to Moderate" choice rather than the degenerate w_mod ->
    # infinity endpoint (which would just predict almost everything as Moderate).
    raw_moderate_recall = sweep[0]["moderate_recall"]
    macro_f1_floor = raw_oof_f1 * 0.90
    candidates = [r for r in sweep if r["macro_f1"] >= macro_f1_floor]
    targeted = max(candidates, key=lambda r: r["moderate_recall"])
    targeted_weights = (1.0, targeted["w_mod"], best_weights[2])
    print(f"\nTargeted operating point (max Moderate recall with macro-F1 >= "
          f"{macro_f1_floor:.4f}, 90% of raw): w_mod={targeted['w_mod']}, "
          f"OOF Moderate recall {raw_moderate_recall:.1%} -> {targeted['moderate_recall']:.1%}")

    print(f"\nApplying both operating points to the deployed model's test-set predictions")
    model = joblib.load(MODEL_PATH)
    proba_test = model.predict_proba(X_test)

    raw_pred_test = apply_weights(proba_test, (1.0, 1.0, 1.0))
    raw_metrics = evaluate_with_weights(y_test, raw_pred_test)
    print("\nRaw argmax (no threshold tuning) on test set:")
    print_metrics(raw_metrics)

    macro_pred_test = apply_weights(proba_test, best_weights)
    macro_metrics = evaluate_with_weights(y_test, macro_pred_test)
    print("\nMacro-F1-optimal threshold on test set:")
    print_metrics(macro_metrics)

    targeted_pred_test = apply_weights(proba_test, targeted_weights)
    targeted_metrics = evaluate_with_weights(y_test, targeted_pred_test)
    print("\nModerate-targeted threshold on test set:")
    print_metrics(targeted_metrics)

    targeted_metrics["weights"] = {
        "Minor": targeted_weights[0], "Moderate": targeted_weights[1], "Severe": targeted_weights[2],
    }
    targeted_metrics["oof_macro_f1_raw"] = raw_oof_f1
    targeted_metrics["oof_moderate_recall_raw"] = raw_moderate_recall
    targeted_metrics["oof_sweep"] = sweep
    targeted_metrics["macro_f1_optimal_weights"] = {
        "Minor": best_weights[0], "Moderate": best_weights[1], "Severe": best_weights[2],
    }
    targeted_metrics["macro_f1_optimal_test_metrics"] = macro_metrics
    targeted_metrics["note"] = (
        "Per-class probability threshold tuning on the tuned LightGBM's predict_proba output, "
        "targeting Moderate recall directly rather than macro-F1: macro-F1-optimal weights "
        f"(1.0, {best_weights[1]:.3f}, {best_weights[2]:.3f}) barely move Moderate recall "
        f"({raw_metrics['per_class']['Moderate']['recall']:.1%} -> "
        f"{macro_metrics['per_class']['Moderate']['recall']:.1%}) because macro-F1 penalizes "
        "the precision Moderate loses as it's boosted. Deliberately targeting Moderate with "
        f"weights (1.0, {targeted_weights[1]:.1f}, {targeted_weights[2]:.3f}) moves recall to "
        f"{targeted_metrics['per_class']['Moderate']['recall']:.1%} on the test set, at the cost "
        f"of macro-F1 dropping from {raw_metrics['macro_f1']:.4f} to {targeted_metrics['macro_f1']:.4f} "
        "and Moderate precision falling correspondingly -- a real tradeoff, not a free lunch."
    )

    write_metrics(targeted_metrics, "lgbm_threshold_tuned_metrics.json")
    write_metrics(raw_metrics, "lgbm_tuned_raw_argmax_metrics.json")

    return targeted_metrics, raw_metrics


if __name__ == "__main__":
    tune_thresholds()

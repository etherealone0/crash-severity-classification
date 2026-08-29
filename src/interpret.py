"""SHAP and permutation-importance interpretability analysis."""

import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, os.path.dirname(__file__))
from train import RANDOM_STATE, load_train_test, split_xy  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_svc.joblib")

# KernelExplainer is model-agnostic but expensive: cost scales with
# explain_rows * background_rows * nsamples. At nsamples=50/background=20 a single
# row took ~3.2s; 200 explained rows against a 50-row background at nsamples=100
# (rather than shap's much larger 'auto' default) keeps this to tens of minutes
# instead of hours, while staying within the task's suggested 200-500 / 50-100 ranges.
SHAP_EXPLAIN_SIZE = 200
SHAP_BACKGROUND_SIZE = 50
SHAP_NSAMPLES = 100

# Plain-English descriptions for the report -- fill in for whichever features end up
# in the top 5, used by print_top_features().
FEATURE_DESCRIPTIONS = {
    "Temperature(F)": "outdoor temperature at crash time",
    "Visibility(mi)": "visibility distance",
    "Wind_Speed(mph)": "wind speed",
    "Precipitation(in)": "rainfall/snowfall amount",
    "Distance(mi)": "length of road segment affected by the crash",
    "Sunrise_Sunset_Night": "crash occurred at night (sunrise/sunset definition)",
    "Civil_Twilight_Night": "crash occurred at night (civil twilight definition)",
    "Hour": "hour of day the crash occurred",
    "State_freq": "how common crashes are in that state (frequency-encoded)",
    "Weather_Condition_freq": "how common that weather condition is (frequency-encoded)",
    "Junction": "crash occurred at a road junction",
    "Traffic_Signal": "crash occurred near a traffic signal",
    "Crossing": "crash occurred near a pedestrian crossing",
    "Amenity": "crash occurred near a roadside amenity",
    "Bump": "crash occurred near a speed bump",
    "Give_Way": "crash occurred near a give-way sign",
    "No_Exit": "crash occurred on a no-exit road",
    "Railway": "crash occurred near a railway crossing",
    "Roundabout": "crash occurred at a roundabout",
    "Station": "crash occurred near a station",
    "Stop": "crash occurred near a stop sign",
    "Traffic_Calming": "crash occurred near a traffic-calming feature",
    "Turning_Loop": "crash occurred at a turning loop",
}
for _day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    FEATURE_DESCRIPTIONS[f"DayOfWeek_{_day}"] = f"crash occurred on a {_day}"


def load_model_and_test():
    model = joblib.load(MODEL_PATH)
    _, test = load_train_test()
    X_test, y_test = split_xy(test)
    return model, X_test, y_test


def compute_shap_values(model, X_test):
    """Explain SHAP_EXPLAIN_SIZE test rows against a SHAP_BACKGROUND_SIZE background,
    using decision_function (one score per class) since this SVC wasn't fit with
    probability=True. Returns (shap_values, explain_sample) where shap_values has
    shape (n_explain, n_features, n_classes)."""
    background = shap.sample(X_test, SHAP_BACKGROUND_SIZE, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(model.decision_function, background)

    explain_sample = X_test.sample(n=SHAP_EXPLAIN_SIZE, random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(explain_sample, nsamples=SHAP_NSAMPLES)
    return shap_values, explain_sample


def rank_features(shap_values, feature_names):
    """Mean absolute SHAP value per feature, averaged across classes and samples."""
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def plot_shap_summary(shap_values, explain_sample, out_path):
    """Overall summary plot: mean |SHAP value| per feature, averaged across classes."""
    mean_abs_per_class = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)
    combined = mean_abs_per_class.mean(axis=1)  # (n_features,)
    order = np.argsort(combined)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([explain_sample.columns[i] for i in order], combined[order], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value| (averaged across classes)")
    ax.set_title("SHAP Feature Importance -- Final Tuned SVC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_shap_analysis():
    """Task 10: rank features by mean absolute SHAP value using KernelExplainer."""
    model, X_test, _ = load_model_and_test()
    print(f"Explaining {SHAP_EXPLAIN_SIZE} rows against a {SHAP_BACKGROUND_SIZE}-row "
          f"background, nsamples={SHAP_NSAMPLES}")

    shap_values, explain_sample = compute_shap_values(model, X_test)
    print("shap_values shape:", shap_values.shape)

    ranked = rank_features(shap_values, list(explain_sample.columns))
    top5 = ranked[:5]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_shap_summary(shap_values, explain_sample, os.path.join(RESULTS_DIR, "shap_summary.png"))
    print(f"Wrote {os.path.join(RESULTS_DIR, 'shap_summary.png')}")

    top_features = [
        {
            "feature": name,
            "mean_abs_shap": score,
            "plain_english": FEATURE_DESCRIPTIONS.get(name, name),
        }
        for name, score in top5
    ]
    out_path = os.path.join(RESULTS_DIR, "top_features.json")
    with open(out_path, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"Wrote {out_path}")

    print("\nTop 5 features by mean |SHAP value|:")
    for i, tf in enumerate(top_features, 1):
        print(f"  {i}. {tf['feature']} ({tf['mean_abs_shap']:.4f}) -- {tf['plain_english']}")

    return top_features


def get_shap_explain_sample(X_test, y_test):
    """Reproduce the exact same 200-row sample used for SHAP, with its true labels."""
    explain_sample = X_test.sample(n=SHAP_EXPLAIN_SIZE, random_state=RANDOM_STATE)
    y_sample = y_test.loc[explain_sample.index]
    return explain_sample, y_sample


def run_permutation_importance():
    """Task 11: cross-check SHAP's ranking with sklearn's permutation_importance on
    the same test subset used for SHAP."""
    from sklearn.inspection import permutation_importance

    model, X_test, y_test = load_model_and_test()
    explain_sample, y_sample = get_shap_explain_sample(X_test, y_test)

    result = permutation_importance(
        model, explain_sample, y_sample, scoring="f1_macro", n_repeats=10, random_state=RANDOM_STATE
    )

    order = np.argsort(result.importances_mean)[::-1]
    ranked = [
        {
            "feature": explain_sample.columns[i],
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in order
    ]
    top5_perm = ranked[:5]

    with open(os.path.join(RESULTS_DIR, "top_features.json")) as f:
        shap_top5 = json.load(f)
    shap_names = [f["feature"] for f in shap_top5]
    perm_names = [f["feature"] for f in top5_perm]
    overlap = sorted(set(shap_names) & set(perm_names))

    output = {
        "top_5": top5_perm,
        "full_ranking": ranked,
        "shap_top_5": shap_names,
        "permutation_top_5": perm_names,
        "agreement_count": len(overlap),
        "agreement_features": overlap,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "permutation_importance.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")

    print("\nPermutation importance top 5 (scoring=f1_macro, n_repeats=10):")
    for i, feat in enumerate(top5_perm, 1):
        print(f"  {i}. {feat['feature']} ({feat['importance_mean']:.4f} +/- {feat['importance_std']:.4f})")
    print(f"\nSHAP top 5:        {shap_names}")
    print(f"Permutation top 5: {perm_names}")
    print(f"Agreement: {len(overlap)}/5 features overlap -> {overlap}")

    return output


if __name__ == "__main__":
    run_shap_analysis()
    print("\n" + "=" * 60 + "\n")
    run_permutation_importance()

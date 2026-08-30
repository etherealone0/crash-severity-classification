"""Streamlit dashboard: confusion matrix, per-class metrics, SHAP summary, top risk factors."""

import json
import os
import sys

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from interpret import SHAP_EXPLAIN_SIZE  # noqa: E402


def load_json(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


st.set_page_config(page_title="Crash Severity Classifier", layout="wide")
st.title("Crash Severity Classification Dashboard")

baseline = load_json("baseline_metrics.json")
smote = load_json("smote_metrics.json")
tuned_svc = load_json("tuned_metrics.json")
lgbm = load_json("lgbm_metrics.json")
lgbm_tuned = load_json("lgbm_tuned_metrics.json")
top_features = load_json("top_features.json")
perm = load_json("permutation_importance.json")

st.caption(
    f"Best model: tuned LightGBM -- accuracy {lgbm_tuned['accuracy']:.1%}, macro-F1 "
    f"{lgbm_tuned['macro_f1']:.1%} on {lgbm_tuned['test_rows']:,} held-out test crashes."
)

st.header("Model Comparison (Ablation)")
variants = [
    ("Baseline (no balancing)", baseline),
    ("SMOTENC-balanced", smote),
    ("Tuned SVC (Optuna)", tuned_svc),
    ("LightGBM (class-weighted)", lgbm),
    ("LightGBM (tuned)", lgbm_tuned),
]
comparison_df = pd.DataFrame(
    [
        {
            "Variant": name,
            "Accuracy": m["accuracy"],
            "Macro-F1": m["macro_f1"],
            "Minor Recall": m["per_class"]["Minor"]["recall"],
            "Moderate Recall": m["per_class"]["Moderate"]["recall"],
            "Severe Recall": m["per_class"]["Severe"]["recall"],
        }
        for name, m in variants
    ]
).set_index("Variant")
st.dataframe(comparison_df.style.format("{:.3f}"))
st.caption(
    "Kept as a full ablation history rather than collapsed to the winning number -- "
    "each row is a deliberate iteration, not a discarded attempt."
)

col1, col2 = st.columns(2)

with col1:
    st.header("Confusion Matrix (Tuned LightGBM)")
    st.image(os.path.join(RESULTS_DIR, "lgbm_confusion_matrix.png"))

with col2:
    st.header("Per-Class Metrics (Tuned LightGBM)")
    per_class_df = pd.DataFrame(lgbm_tuned["per_class"]).T
    per_class_df = per_class_df.rename(
        columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"}
    )
    per_class_df["Support"] = per_class_df["Support"].astype(int)
    st.dataframe(per_class_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}))

col3, col4 = st.columns(2)

with col3:
    st.header("SHAP Feature Importance")
    st.image(os.path.join(RESULTS_DIR, "shap_summary.png"))

with col4:
    st.header("Top 5 Crash Risk Factors")
    for i, feat in enumerate(top_features, 1):
        st.markdown(f"**{i}. {feat['plain_english'].capitalize()}** ({feat['feature']})")
    st.caption(
        f"Ranked by exact TreeExplainer SHAP values on a {SHAP_EXPLAIN_SIZE:,}-row test sample "
        f"of the tuned LightGBM model; cross-checked against permutation importance "
        f"({perm['agreement_count']}/5 top features agree)."
    )

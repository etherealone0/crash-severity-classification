"""Streamlit dashboard: confusion matrix, per-class metrics, SHAP summary, top risk factors."""

import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_json(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


st.set_page_config(page_title="Crash Severity Classifier", layout="wide")
st.title("Crash Severity Classification Dashboard")

tuned = load_json("tuned_metrics.json")
top_features = load_json("top_features.json")

st.caption(
    f"Final tuned SVC -- accuracy {tuned['accuracy']:.1%}, macro-F1 {tuned['macro_f1']:.1%} "
    f"on {tuned['test_rows']:,} held-out test crashes."
)

col1, col2 = st.columns(2)

with col1:
    st.header("Confusion Matrix")
    st.image(os.path.join(RESULTS_DIR, "confusion_matrix.png"))

with col2:
    st.header("Per-Class Metrics")
    per_class_df = pd.DataFrame(tuned["per_class"]).T
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
        "Ranked by mean absolute SHAP value on a 200-row test sample; "
        "cross-checked against permutation importance (see results/permutation_importance.json)."
    )

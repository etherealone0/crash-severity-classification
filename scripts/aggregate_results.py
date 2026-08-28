"""Read every results/*.json and write one results/summary.md with every number
referenced by the resume bullets, so nobody has to hunt through separate files."""

import json
import os
import sys

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from interpret import SHAP_EXPLAIN_SIZE  # noqa: E402


def load_json(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


def dataset_summary():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    combined = pd.concat([train["severity_class"], test["severity_class"]])
    return len(combined), combined.value_counts()


def comparison_table(baseline, smote, tuned):
    rows = [("Baseline (no balancing)", baseline), ("SMOTENC-balanced", smote), ("Tuned (Optuna)", tuned)]
    lines = [
        "| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        pc = m["per_class"]
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{pc['Minor']['recall']:.3f} | {pc['Moderate']['recall']:.3f} | {pc['Severe']['recall']:.3f} |"
        )
    return "\n".join(lines)


def main():
    baseline = load_json("baseline_metrics.json")
    smote = load_json("smote_metrics.json")
    tuned = load_json("tuned_metrics.json")
    top_features = load_json("top_features.json")
    perm = load_json("permutation_importance.json")

    n_total, class_counts = dataset_summary()
    table = comparison_table(baseline, smote, tuned)

    severe_recall_baseline = baseline["per_class"]["Severe"]["recall"]
    severe_recall_smote = smote["per_class"]["Severe"]["recall"]
    macro_f1_smote = smote["macro_f1"]
    macro_f1_tuned = tuned["macro_f1"]

    class_lines = "\n".join(f"  - {cls}: {count:,}" for cls, count in class_counts.items())

    shap_lines = ["| Rank | Feature | Mean \\|SHAP\\| | Plain English |", "|---|---|---|---|"]
    for i, tf in enumerate(top_features, 1):
        shap_lines.append(f"| {i} | {tf['feature']} | {tf['mean_abs_shap']:.4f} | {tf['plain_english']} |")

    report = f"""# Project Summary

Every number referenced by the 4 resume bullets, in one place.

## Dataset

- Total modeling rows (train + test, post-dedup): {n_total:,}
- Class distribution:
{class_lines}

## Model comparison (test set, all three trained/evaluated the same way)

{table}

## Key deltas

- **Bullet 1 (headline):** macro-F1 {macro_f1_tuned:.1%} across 3 severity classes on {n_total:,}+ crash records
- **Bullet 2 (SMOTENC):** Severe-class recall {severe_recall_baseline:.1%} (baseline) -> {severe_recall_smote:.1%} (SMOTENC)
- **Bullet 3 (tuning):** macro-F1 {macro_f1_smote:.1%} (Task 7, default params) -> {macro_f1_tuned:.1%} (Task 8, tuned), via Optuna over {tuned['n_trials']} trials of {tuned['cv_folds']}-fold CV (best params: {tuned['best_params']})
- **Bullet 4 (interpretability):** top feature is {top_features[0]['feature']} ({top_features[0]['plain_english']}), across {SHAP_EXPLAIN_SIZE} test predictions, validated against permutation importance ({perm['agreement_count']}/5 top features agree)

## Top 5 SHAP features

{chr(10).join(shap_lines)}

## Permutation importance cross-check

- SHAP top 5: {perm['shap_top_5']}
- Permutation top 5: {perm['permutation_top_5']}
- Agreement: {perm['agreement_count']}/5 features overlap -> {perm['agreement_features']}
"""

    out_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

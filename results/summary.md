# Project Summary

Every number referenced by the resume bullets, in one place. Kept as a full
ablation history (baseline -> SMOTENC -> tuned SVC -> LightGBM -> tuned
LightGBM) rather than just the winning number, since the iteration story is
itself part of what this project demonstrates.

## Dataset

- Total modeling rows (train + test, post-dedup): 1,943,711
- Class distribution:
  - Minor: 1,558,464
  - Moderate: 335,401
  - Severe: 49,846

## Model comparison (test set, all five trained/evaluated the same way)

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.802 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.374 | 0.291 | 0.299 | 0.732 | 0.296 |
| Tuned (Optuna) | 0.657 | 0.387 | 0.711 | 0.499 | 0.025 |
| LightGBM (class-weighted) | 0.550 | 0.442 | 0.496 | 0.759 | 0.824 |
| LightGBM (tuned) | 0.786 | 0.563 | 0.837 | 0.610 | 0.389 |

## Key deltas

- **Headline:** best model is tuned LightGBM at 56.3% macro-F1 across 3 severity classes on 1,943,711+ crash records
- **SMOTENC:** Severe-class recall 0.0% (baseline) -> 29.6% (SMOTENC)
- **SVC tuning:** macro-F1 29.1% (default params) -> 38.7% (tuned), via Optuna over 8 trials of 3-fold CV (best params: {'C': 21.368329072358772, 'gamma': 0.0011526449540315614})
- **Tree ensemble:** LightGBM, trained on the full 1,554,968-row split with class-weighted balancing instead of SMOTENC, reaches 44.2% macro-F1 and 82.4% Severe recall -- beating every SVC variant on both at once
- **Tree ensemble tuning:** Optuna over 60 trials tuning n_estimators/max_depth/learning_rate plus per-class weighting pushes macro-F1 to 56.3% (CV score 54.9%, so no leakage-driven inflation)
- **Interpretability:** top feature is Distance(mi) (length of road segment affected by the crash), via exact TreeExplainer SHAP values on 20,000 test predictions, validated against permutation importance (4/5 top features agree)
- **Targeting Moderate directly:** macro-F1-optimal thresholds barely move Moderate recall, but deliberately boosting Moderate's decision weight moves recall 61.0% -> 81.3%, at the cost of macro-F1 dropping to 53.7% -- a real, explicit tradeoff rather than a free win

## Top 5 SHAP features (tuned LightGBM, TreeExplainer)

| Rank | Feature | Mean \|SHAP\| | Plain English |
|---|---|---|---|
| 1 | Distance(mi) | 0.7815 | length of road segment affected by the crash |
| 2 | State_freq | 0.2752 | how common crashes are in that state (frequency-encoded) |
| 3 | Weather_Condition_freq | 0.1293 | how common that weather condition is (frequency-encoded) |
| 4 | Hour | 0.1054 | hour of day the crash occurred |
| 5 | Wind_Speed(mph) | 0.0963 | wind speed |

## Permutation importance cross-check

- SHAP top 5: ['Distance(mi)', 'State_freq', 'Weather_Condition_freq', 'Hour', 'Wind_Speed(mph)']
- Permutation top 5: ['Distance(mi)', 'State_freq', 'Weather_Condition_freq', 'Wind_Speed(mph)', 'Traffic_Signal']
- Agreement: 4/5 features overlap -> ['Distance(mi)', 'State_freq', 'Weather_Condition_freq', 'Wind_Speed(mph)']

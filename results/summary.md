# Project Summary

Every number referenced by the 4 resume bullets, in one place.

## Dataset

- Total modeling rows (train + test, post-dedup): 397,358
- Class distribution:
  - Minor: 319,696
  - Moderate: 67,214
  - Severe: 10,448

## Model comparison (test set, all three trained/evaluated the same way)

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.805 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.187 | 0.176 | 0.111 | 0.488 | 0.562 |
| Tuned (Optuna) | 0.701 | 0.350 | 0.835 | 0.163 | 0.056 |

## Key deltas

- **Bullet 1 (headline):** macro-F1 35.0% across 3 severity classes on 397,358+ crash records
- **Bullet 2 (SMOTENC):** Severe-class recall 0.0% (baseline) -> 56.2% (SMOTENC)
- **Bullet 3 (tuning):** macro-F1 17.6% (Task 7, default params) -> 35.0% (Task 8, tuned), via Optuna over 20 trials of 3-fold CV (best params: {'C': 1.9151000143366523, 'gamma': 0.4920709465227078})
- **Bullet 4 (interpretability):** top feature is Temperature(F) (outdoor temperature at crash time), across 200 test predictions, validated against permutation importance (3/5 top features agree)

## Top 5 SHAP features

| Rank | Feature | Mean \|SHAP\| | Plain English |
|---|---|---|---|
| 1 | Temperature(F) | 0.1224 | outdoor temperature at crash time |
| 2 | Hour | 0.1133 | hour of day the crash occurred |
| 3 | Wind_Speed(mph) | 0.1039 | wind speed |
| 4 | DayOfWeek_Friday | 0.0399 | crash occurred on a Friday |
| 5 | Visibility(mi) | 0.0392 | visibility distance |

## Permutation importance cross-check

- SHAP top 5: ['Temperature(F)', 'Hour', 'Wind_Speed(mph)', 'DayOfWeek_Friday', 'Visibility(mi)']
- Permutation top 5: ['Wind_Speed(mph)', 'Temperature(F)', 'Hour', 'DayOfWeek_Saturday', 'DayOfWeek_Monday']
- Agreement: 3/5 features overlap -> ['Hour', 'Temperature(F)', 'Wind_Speed(mph)']

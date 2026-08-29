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
| SMOTENC-balanced | 0.301 | 0.251 | 0.217 | 0.674 | 0.459 |
| Tuned (Optuna) | 0.679 | 0.394 | 0.751 | 0.431 | 0.042 |

## Key deltas

- **Bullet 1 (headline):** macro-F1 39.4% across 3 severity classes on 397,358+ crash records
- **Bullet 2 (SMOTENC):** Severe-class recall 0.0% (baseline) -> 45.9% (SMOTENC)
- **Bullet 3 (tuning):** macro-F1 25.1% (Task 7, default params) -> 39.4% (Task 8, tuned), via Optuna over 8 trials of 3-fold CV (best params: {'C': 21.368329072358772, 'gamma': 0.0011526449540315614})
- **Bullet 4 (interpretability):** top feature is DayOfWeek_Friday (crash occurred on a Friday), across 200 test predictions, validated against permutation importance (3/5 top features agree)

## Top 5 SHAP features

| Rank | Feature | Mean \|SHAP\| | Plain English |
|---|---|---|---|
| 1 | DayOfWeek_Friday | 0.2272 | crash occurred on a Friday |
| 2 | DayOfWeek_Thursday | 0.1810 | crash occurred on a Thursday |
| 3 | Distance(mi) | 0.1806 | length of road segment affected by the crash |
| 4 | DayOfWeek_Wednesday | 0.1704 | crash occurred on a Wednesday |
| 5 | DayOfWeek_Saturday | 0.1599 | crash occurred on a Saturday |

## Permutation importance cross-check

- SHAP top 5: ['DayOfWeek_Friday', 'DayOfWeek_Thursday', 'Distance(mi)', 'DayOfWeek_Wednesday', 'DayOfWeek_Saturday']
- Permutation top 5: ['Distance(mi)', 'Weather_Condition_freq', 'DayOfWeek_Wednesday', 'Hour', 'DayOfWeek_Saturday']
- Agreement: 3/5 features overlap -> ['DayOfWeek_Saturday', 'DayOfWeek_Wednesday', 'Distance(mi)']

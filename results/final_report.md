# Final Model Evaluation

Final tuned SVC (`models/final_svc.joblib`), evaluated on the held-out test set
(79472 rows).

## Model comparison (test set, all four trained/evaluated the same way)

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.805 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.301 | 0.251 | 0.217 | 0.674 | 0.459 |
| Tuned (Optuna) | 0.679 | 0.394 | 0.751 | 0.431 | 0.042 |
| LightGBM (class-weighted) | 0.552 | 0.443 | 0.500 | 0.761 | 0.816 |

Severe-class recall goes from 0.0% (baseline) to
45.9% (SMOTENC) to 4.2% (tuned): Optuna
optimizes mean macro-F1 across CV folds, which favors getting Minor and Moderate
right and trades away most of SMOTENC's Severe-recall gain to do it.

LightGBM (`models/final_lgbm.joblib`), trained on the full training split with
`class_weight="balanced"` instead of SMOTENC, beats every SVC variant on macro-F1
(0.443) and Severe recall (81.6%) at once --
both the scaling fix (no subsampling) and the tree model's better fit on this
mixed categorical/continuous data show up directly in the numbers.

## Final (tuned) model -- per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Minor | 0.841 | 0.751 | 0.794 | 63939 |
| Moderate | 0.276 | 0.431 | 0.336 | 13443 |
| Severe | 0.066 | 0.042 | 0.051 | 2090 |

## Confusion matrix

![Confusion Matrix](confusion_matrix.png)

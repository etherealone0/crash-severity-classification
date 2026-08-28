# Final Model Evaluation

Final tuned SVC (`models/final_svc.joblib`), evaluated on the held-out test set
(79472 rows).

## Model comparison (test set, all three trained/evaluated the same way)

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.805 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.187 | 0.176 | 0.111 | 0.488 | 0.562 |
| Tuned (Optuna) | 0.612 | 0.370 | 0.662 | 0.457 | 0.065 |

Severe-class recall goes from 0.0% (baseline) to
56.2% (SMOTENC) to 6.5% (tuned): Optuna
optimizes mean macro-F1 across CV folds, which favors getting Minor and Moderate
right and trades away most of SMOTENC's Severe-recall gain to do it.

## Final (tuned) model -- per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Minor | 0.837 | 0.662 | 0.740 | 63939 |
| Moderate | 0.229 | 0.457 | 0.305 | 13443 |
| Severe | 0.064 | 0.065 | 0.064 | 2090 |

## Confusion matrix

![Confusion Matrix](confusion_matrix.png)

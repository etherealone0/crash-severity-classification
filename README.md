# Crash Severity Classification

Predicts crash severity (Minor / Moderate / Severe) from weather, road-feature,
and time-of-day data. Compares five model variants -- baseline SVC, SMOTENC-
balanced SVC, Optuna-tuned SVC, class-weighted LightGBM, and Optuna-tuned
LightGBM -- as a deliberate ablation history, not just a single winning number.
The tuned LightGBM is both the deployed model and the one SHAP/permutation-
importance interpretability explains. Results are served through a Streamlit
dashboard.

Dataset: [US Accidents (2016-2023) by Sobhan Moosavi](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
(~7.7M rows, 46 columns), stratify-sampled down to 2M rows for local iteration.

## Highlights

- Built and benchmarked 5 crash-severity classifier variants (SVC baseline/SMOTENC/Optuna-tuned,
  LightGBM class-weighted/Optuna-tuned) on 1.9M+ crash records, with the best model -- tuned
  LightGBM -- reaching 56.3% macro-F1 across 3 severity classes, a 91% relative improvement over
  the untuned baseline.
- Compared two class-imbalance strategies head-to-head -- SMOTENC oversampling vs. class-weighted
  loss -- lifting Severe-crash recall from 0% (baseline) to 82.4% (class-weighted LightGBM); also
  ran targeted per-class threshold tuning to move Moderate-class recall from 61% to 81% on demand,
  with the precision tradeoff made explicit rather than hidden behind an averaged metric.
- Ran two leak-free Optuna hyperparameter searches (SVC and LightGBM) using `cross_val_score`
  inside an `imblearn.Pipeline` to eliminate resampling leakage, improving LightGBM macro-F1 27%
  (44.2% -> 56.3%) with cross-validation and held-out test scores within 1.4 points of each other,
  confirming the tuning signal was real.
- Replaced an approximate, ~2-hour SHAP `KernelExplainer` run (200 samples) with exact
  `TreeExplainer` values on 20,000 samples in seconds, identifying crash-affected road distance as
  the dominant predictor and cross-validating the top-5 feature ranking against permutation
  importance (4/5 agreement).

## Setup

```bash
python -m venv venv
./venv/Scripts/activate   # venv\Scripts\activate on Windows cmd, or source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

Then get the dataset onto disk. Either:

- **Kaggle API**: put your credentials at `~/.kaggle/kaggle.json` (or set
  `KAGGLE_USERNAME`/`KAGGLE_KEY`) and `src/data_prep.py` will download it
  automatically via `kagglehub`, or
- **Manual**: download the CSV from the link above and place it anywhere under
  `data/raw/`.

## Reproducing the results end to end

Run in order -- each stage depends on the previous one's output:

```bash
python src/data_prep.py            # sample, clean, dedup, split -> data/processed/{train,test}.parquet
python src/train.py                # baseline -> SMOTENC -> final consolidated report
python src/tune.py                 # Optuna search (SVC) -> models/final_svc.joblib
python src/train_ensemble.py       # LightGBM (class-weighted) -> models/final_lgbm.joblib
python src/tune_ensemble.py        # Optuna search (LightGBM) -> models/final_lgbm_tuned.joblib
python src/threshold_tune.py       # per-class threshold tuning targeting Moderate recall
python src/interpret.py            # SHAP (TreeExplainer) + permutation importance
python scripts/aggregate_results.py # everything -> results/summary.md
streamlit run dashboard/app.py     # view it
```

`results/summary.md` alone has every number below.

## The severity-class mapping

The raw dataset's `Severity` column has 4 levels (1-4), but this project
collapses it to 3 classes -- **Minor** ({1, 2}), **Moderate** ({3}), and
**Severe** ({4}) -- to match the intended framing. Severity 1 is folded into
Minor because it's under 1% of rows and represents the same real-world outcome
(little to no traffic impact) as Severity 2; keeping it as its own class would
just be a near-empty fourth bucket. Moderate and Severe are left as their own
classes since they represent meaningfully different outcomes (significant delay
vs. major/most disruptive). The resulting classes are still heavily imbalanced
(roughly 80% / 17% / 3%), which SMOTENC and class-weighting each address below.

## Feature set

30 features (see the comment block above `select_and_clean_features` in
`src/data_prep.py` for the full kept/dropped rationale): 5 weather/distance
numerics (including `Distance(mi)`, the length of road segment affected by
the crash), 13 boolean road-feature/POI flags, 2 lighting flags
(`Sunrise_Sunset_Night`, `Civil_Twilight_Night`), `Hour`, 7 one-hot
`DayOfWeek_*` columns, and 2 frequency-encoded columns (`State`,
`Weather_Condition`).

## Results

Evaluated on the held-out test set (388,743 rows), on a 1,943,711-row modeling
dataset (2M sampled from the full ~7.7M-row dataset, minus 56,292
near-duplicate crash records):

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.802 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.374 | 0.291 | 0.299 | 0.732 | 0.296 |
| Tuned SVC (Optuna) | 0.657 | 0.387 | 0.711 | 0.499 | 0.025 |
| LightGBM (class-weighted) | 0.550 | 0.442 | 0.496 | 0.759 | 0.824 |
| **LightGBM (tuned)** | **0.786** | **0.563** | 0.837 | 0.610 | 0.389 |

- SMOTENC alone takes Severe-class recall from 0% to 29.6%, at the cost of
  overall macro-F1 (an untuned SVC overcorrects toward the minority classes).
- Optuna tuning of the SVC (8 trials, 3-fold CV, objective evaluated honestly
  -- see "CV leakage fix" below) recovers macro-F1 to 0.387, better than either
  untuned SVC variant, but trades away most of SMOTENC's Severe-recall gain to
  get there (0.025 vs. 0.296) -- a real precision/recall tension worth knowing
  about rather than hiding.
- LightGBM (class-weighted, trained on the full training split -- see
  "Tree ensemble" below) beats every SVC variant on both macro-F1 and Severe
  recall at the same time, with no precision/recall tradeoff to apologize for.
- Tuning LightGBM itself (60 trials over n_estimators/max_depth/learning_rate
  plus per-class weighting, same leak-free CV discipline as the SVC study)
  pushes macro-F1 to 0.563 -- the best result of any variant, and the model
  actually deployed and interpreted below.
- Top 5 SHAP features (tuned LightGBM, TreeExplainer): **Distance(mi)**,
  **State_freq**, **Weather_Condition_freq**, **Hour**, **Wind_Speed(mph)**.
  `Distance(mi)` is the clear #1 by a wide margin (mean |SHAP| 0.78 vs. 0.28
  for #2), and is also the #1 feature by permutation importance -- 4 of 5
  features agree between the two methods (see "Interpretability" below).

## CV leakage fix (and why the numbers moved)

An earlier version of `src/tune.py` applied SMOTENC once to the whole training
subsample, then ran cross-validation on top of that already-resampled data.
That leaks information: synthetic minority points generated from what became a
validation fold's real neighbors can end up in the training fold (and vice
versa), so folds aren't independent. The symptom was a CV macro-F1 during the
search (~0.85) wildly higher than the honest held-out test score (0.350).

The fix: `src/tune.py` now wraps SMOTENC and the SVC in an
`imblearn.pipeline.Pipeline` and runs `cross_val_score` on the *original,
imbalanced* training subsample. That pipeline gets cloned and refit inside
`cross_val_score` for every fold, so SMOTENC only ever sees that fold's
training rows -- validation folds stay real and untouched. After the fix,
Optuna's best CV score (0.382) lands close to the true test score (0.387),
which is the actual signal a leak-free search should produce.

Two costs came with the fix, both handled and documented in `src/tune.py`:
- Refitting SMOTENC per fold instead of once is 5-6x slower per trial, so the
  trial count dropped from 20 to 8 to stay tractable.
- A few (low-`C`, low-`gamma`) corners of the search make libsvm's solver
  converge extremely slowly on this data -- one uncapped fit ran 2+ hours
  before finishing at a mediocre score anyway. `SVC(..., max_iter=20_000)`
  bounds worst-case time per fit without changing which regions score well.

`src/tune_ensemble.py` follows the same discipline for LightGBM: cross-
validation runs on the original imbalanced data with `cross_val_score`, honest
end to end. There's no resampler in that pipeline to leak across folds in the
first place (class-weighting replaces SMOTENC for LightGBM), so the risk this
guards against is milder there -- but the CV score (0.549) still lands close
to the test score (0.563), confirming the search wasn't fooling itself either
way.

## Notes on scale

`sklearn.svm.SVC(kernel="rbf")` doesn't scale to hundreds of thousands of rows
in practical time, especially once SMOTENC balances the classes (that makes the
separation problem genuinely harder for libsvm). `src/train.py` and
`src/tune.py` stratify-subsample the training split down to 8,000 rows for the
SVC variants, while always evaluating on the full, untouched test set.

LightGBM doesn't share that ceiling: `src/train_ensemble.py` trains on the
full 1,554,968-row training split directly, and `src/tune_ensemble.py`
searches hyperparameters on a 200,000-row subsample for speed (a single
3-fold CV pass over the full split took ~2 minutes per trial -- too slow for
a 60-trial search) before refitting the final model on the full split.

## Tree ensemble (`src/train_ensemble.py`, `src/tune_ensemble.py`)

SVC's subsampling isn't just an inconvenience -- it throws away most of the
training data, and the RBF kernel is a mediocre fit for tabular data that's
mostly one-hot/frequency-encoded categoricals to begin with. `LGBMClassifier`
doesn't share either problem: gradient-boosted trees split natively on
categorical-flavored features and don't have SVC's O(n^2-n^3) scaling wall.

`train_lgbm()` trains a default-hyperparameter LightGBM with
`class_weight="balanced"` on the full training split as a direct alternative
to SMOTENC for handling the imbalance -- and already beats every SVC variant.
`train_tuned_lgbm()` then runs a 60-trial Optuna study (leak-free CV, same
principle as the SVC fix) over `n_estimators`, `max_depth`, `learning_rate`,
and per-class weight multipliers, refits the winning configuration on the full
1,554,968-row split, and pushes macro-F1 to 0.563 -- the best result of any
variant. This tuned model (`models/final_lgbm_tuned.joblib`) is the one
deployed model: it's what the dashboard displays and what SHAP explains.

## Targeting Moderate recall directly (`src/threshold_tune.py`)

Severity is ordinal (1 < 2 < 3 < 4), so the naive expectation is that Moderate
gets confused with its ordinal neighbor, Severe. The confusion matrix on the
tuned LightGBM doesn't show that pattern: Moderate is misclassified mostly
*into Minor* (33.5% of its rows) rather than into Severe (5.5%), and Severe is
misclassified mostly into Minor too (56.0% vs. 5.1% into Moderate). That's
majority-class pull (Minor is 80% of the data), not ordinal-adjacency
confusion -- worth correcting before "fixing" the wrong problem.

Per-class probability threshold tuning counters majority-class pull directly:
multiply each class's predicted probability by a weight before taking argmax,
so Moderate/Severe don't need to out-score Minor's raw probability to win.
Weights are grid-searched on out-of-fold predictions from the training set
(3-fold CV, same hyperparameters as the deployed model) so the search never
touches the test set, then applied once to the deployed model's test-set
probabilities.

Two results worth separating:
- **Macro-F1-optimal weights** barely move Moderate recall at all (61.0% ->
  61.7%) -- macro-F1 penalizes the precision Moderate loses as it's boosted
  by almost exactly as much as it rewards the recall gained, so the "best
  average" is close to doing nothing. This is exactly the failure mode of
  optimizing an average instead of the metric you actually care about.
- **Deliberately targeting Moderate** (boosting its weight to 2.0x) moves
  recall to 81.3% on the test set, at a real, explicit cost: macro-F1 drops
  from 0.563 to 0.537 and Minor recall falls from 83.7% to 72.6%. Whether
  that tradeoff is worth it is a product decision, not a modeling one -- the
  point is having the number to make that decision with.

## Interpretability (`src/interpret.py`)

SHAP and permutation importance both explain the deployed model: tuned
LightGBM. An earlier version of this pipeline used `shap.KernelExplainer`
against the SVC's `decision_function`, which is model-agnostic but expensive
(cost scales with explain-rows x background-rows x Monte-Carlo samples) --
200 explained rows against a 50-row background took roughly 1.5-2 hours.
`shap.TreeExplainer` computes *exact* SHAP values directly from the tree
structure, no background sample or Monte-Carlo approximation needed, and is
fast enough to explain 20,000 rows in seconds.

Top 5 SHAP features: `Distance(mi)` (mean |SHAP| 0.78), `State_freq` (0.28),
`Weather_Condition_freq` (0.13), `Hour` (0.11), `Wind_Speed(mph)` (0.10).
Cross-checked against `sklearn.inspection.permutation_importance` on the same
20,000-row sample (`scoring="f1_macro"`, 10 repeats): 4 of 5 features agree
(`Distance(mi)`, `State_freq`, `Weather_Condition_freq`, `Wind_Speed(mph)`),
with `Distance(mi)` ranked #1 by both methods independently.

## Project structure

```
data/raw/               raw + sampled CSVs (gitignored)
data/processed/         train/test parquet + feature matrix (gitignored)
src/data_prep.py        download, sample, clean, dedup, split
src/train.py            baseline, SMOTENC, final consolidated evaluation
src/tune.py             Optuna hyperparameter search (SVC)
src/train_ensemble.py   LightGBM (class-weighted) tree-ensemble variant
src/tune_ensemble.py    Optuna hyperparameter search (LightGBM)
src/threshold_tune.py   per-class probability threshold tuning
src/interpret.py        SHAP (TreeExplainer) + permutation importance
scripts/                results aggregation
models/                 final_svc.joblib, final_lgbm.joblib, final_lgbm_tuned.joblib
                        (gitignored, regenerate via the scripts above)
results/                all metrics, plots, and the final summary.md
dashboard/app.py        Streamlit results dashboard
```

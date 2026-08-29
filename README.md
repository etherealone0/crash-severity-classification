# Crash Severity Classification

Predicts crash severity (Minor / Moderate / Severe) from weather, road-feature,
and time-of-day data using an SVC classifier, with SMOTENC class balancing,
Optuna hyperparameter tuning, and SHAP/permutation-importance interpretability.
Results are served through a Streamlit dashboard.

Dataset: [US Accidents (2016-2023) by Sobhan Moosavi](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
(~7.7M rows, 46 columns), stratify-sampled down to 400K rows for local iteration.

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
python src/tune.py                 # Optuna search -> models/final_svc.joblib
python src/interpret.py            # SHAP + permutation importance
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
(roughly 80% / 17% / 3%), which is the imbalance Tasks 6-7 address.

## Feature set

30 features (see the comment block above `select_and_clean_features` in
`src/data_prep.py` for the full kept/dropped rationale): 5 weather/distance
numerics (including `Distance(mi)`, the length of road segment affected by
the crash), 13 boolean road-feature/POI flags, 2 lighting flags
(`Sunrise_Sunset_Night`, `Civil_Twilight_Night`), `Hour`, 7 one-hot
`DayOfWeek_*` columns, and 2 frequency-encoded columns (`State`,
`Weather_Condition`).

## Results

Final tuned SVC, evaluated on the held-out test set (79,472 rows), on a
397,358-row modeling dataset (400K sampled from the full ~7.7M-row dataset,
minus 2,642 near-duplicate crash records):

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.805 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.301 | 0.251 | 0.217 | 0.674 | 0.459 |
| Tuned (Optuna) | 0.679 | 0.394 | 0.751 | 0.431 | 0.042 |

- SMOTENC alone takes Severe-class recall from 0% to 45.9%, at the cost of
  overall macro-F1 (an untuned SVC overcorrects toward the minority classes).
- Optuna tuning (8 trials, 3-fold CV, objective evaluated honestly -- see
  "CV leakage fix" below) recovers macro-F1 to 0.394, better than either
  untuned variant, but trades away most of SMOTENC's Severe-recall gain to get
  there (0.042 vs. 0.459) -- a real precision/recall tension worth knowing
  about rather than hiding.
- Top 5 SHAP features: **DayOfWeek_Friday**, **DayOfWeek_Thursday**,
  **Distance(mi)**, **DayOfWeek_Wednesday**, **DayOfWeek_Saturday** -- this
  particular tuned model (a low `gamma=0.00115`, very smooth decision boundary)
  leans heavily on day-of-week, but `Distance(mi)` (added along with the
  lighting flags and a check that all POI flags were already present) lands
  a strong #3, and is the single #1 feature by permutation importance.
  Cross-checked with permutation importance: 3 of 5 features agree
  (`DayOfWeek_Saturday`, `DayOfWeek_Wednesday`, `Distance(mi)`).

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
Optuna's best CV score (0.388) lands close to the true test score (0.394),
which is the actual signal a leak-free search should produce.

Two costs came with the fix, both handled and documented in `src/tune.py`:
- Refitting SMOTENC per fold instead of once is 5-6x slower per trial, so the
  trial count dropped from 20 to 8 to stay tractable.
- A few (low-`C`, low-`gamma`) corners of the search make libsvm's solver
  converge extremely slowly on this data -- one uncapped fit ran 2+ hours
  before finishing at a mediocre score anyway. `SVC(..., max_iter=20_000)`
  bounds worst-case time per fit without changing which regions score well.

## Notes on scale

`sklearn.svm.SVC(kernel="rbf")` doesn't scale to hundreds of thousands of rows
in practical time, especially once SMOTENC balances the classes (that makes the
separation problem genuinely harder for libsvm). `src/train.py` stratify-
subsamples the training split down to 8,000 rows before training/tuning, while
always evaluating on the full, untouched test set.

## Project structure

```
data/raw/           raw + sampled CSVs (gitignored)
data/processed/     train/test parquet + feature matrix (gitignored)
src/data_prep.py    download, sample, clean, dedup, split
src/train.py        baseline, SMOTENC, final consolidated evaluation
src/tune.py         Optuna hyperparameter search
src/interpret.py    SHAP + permutation importance
scripts/            results aggregation
models/             final_svc.joblib (gitignored, regenerate via src/tune.py)
results/            all metrics, plots, and the final summary.md
dashboard/app.py    Streamlit results dashboard
```

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

## Results

Final tuned SVC, evaluated on the held-out test set (79,472 rows), on a
397,358-row modeling dataset (400K sampled from the full ~7.7M-row dataset,
minus 2,642 near-duplicate crash records):

| Variant | Accuracy | Macro-F1 | Minor Recall | Moderate Recall | Severe Recall |
|---|---|---|---|---|---|
| Baseline (no balancing) | 0.805 | 0.297 | 1.000 | 0.000 | 0.000 |
| SMOTENC-balanced | 0.187 | 0.176 | 0.111 | 0.488 | 0.562 |
| Tuned (Optuna) | 0.701 | 0.350 | 0.835 | 0.163 | 0.056 |

- SMOTENC alone takes Severe-class recall from 0% to 56.2%, at the cost of
  overall macro-F1 (an untuned SVC overcorrects toward the minority classes).
- Optuna tuning (20 trials, 3-fold CV) recovers macro-F1 to 0.350 -- better
  than either untuned variant -- but trades away most of SMOTENC's Severe-recall
  gain to get there, a real precision/recall tension worth knowing about rather
  than hiding.
- Top 5 SHAP features: **Temperature(F)**, **Hour**, **Wind_Speed(mph)**,
  **DayOfWeek_Friday**, **Visibility(mi)** -- weather and time-of-day dominate
  over road-feature flags. Cross-checked with permutation importance: the top 3
  match exactly (just in a different order).

## Notes on scale

`sklearn.svm.SVC(kernel="rbf")` doesn't scale to hundreds of thousands of rows
in practical time, especially once SMOTENC balances the classes (that makes the
separation problem genuinely harder for libsvm). `src/train.py` stratify-
subsamples the training split down to 8,000 rows before training/tuning, while
always evaluating on the full, untouched test set. Separately, the Optuna CV
score during tuning (~0.85) is optimistic: it's computed on folds carved out of
already-SMOTENC-resampled data, where synthetic minority points can leak
similarity across folds. The test-set macro-F1 (0.350) is the trustworthy
number and is what's reported above.

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

"""Data acquisition, cleaning, feature engineering, and train/test splitting."""

import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAMPLE_PATH = os.path.join(RAW_DIR, "us_accidents_sample.csv")
KAGGLE_DATASET = "sobhanmoosavi/us-accidents"
SAMPLE_SIZE = 400_000
RANDOM_STATE = 42


def find_raw_csv():
    """Locate the full US Accidents CSV under data/raw/, however it got there."""
    candidates = sorted(
        f
        for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if os.path.basename(f) != os.path.basename(SAMPLE_PATH)
    )
    return candidates[0] if candidates else None


def has_kaggle_credentials():
    """Check for Kaggle API credentials without triggering any interactive auth flow."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return os.path.exists(os.path.expanduser(os.path.join("~", ".kaggle", "kaggle.json")))


def download_raw_csv():
    """kagglehub download; only called once credentials are confirmed present."""
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csvs = glob.glob(os.path.join(path, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")
    return csvs[0]


def get_raw_csv_path():
    """Return a path to the full raw CSV, downloading it only if credentials are configured."""
    existing = find_raw_csv()
    if existing:
        return existing

    manual_instructions = (
        "No raw CSV found under data/raw/. Download 'US Accidents (2016-2023)' by "
        "Sobhan Moosavi from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents "
        "and place the CSV under data/raw/."
    )

    if not has_kaggle_credentials():
        raise FileNotFoundError(manual_instructions)

    try:
        return download_raw_csv()
    except Exception as exc:
        raise FileNotFoundError(f"{manual_instructions} (kagglehub download also failed: {exc})") from exc


def stratified_sample(df, n=SAMPLE_SIZE, random_state=RANDOM_STATE):
    """Sample n rows stratified by (State, Severity) so no region or class is dropped."""
    frac = min(n / len(df), 1.0)
    sample = df.groupby(["State", "Severity"], group_keys=False).sample(
        frac=frac, random_state=random_state
    )
    return sample.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_sampled_data():
    """Load the stratified sample, building it from the raw CSV on first run."""
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH)

    raw_path = get_raw_csv_path()
    print(f"Reading raw CSV: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Full dataset shape: {df.shape}")

    sample = stratified_sample(df)
    os.makedirs(RAW_DIR, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote stratified sample to {SAMPLE_PATH}")
    return sample


# The dataset's native Severity is 1-4, but we collapse it to 3 classes to match
# the reference project's framing. Severity 1 is a tiny sliver of the data (under
# 1% of rows) and represents the same "minor" real-world outcome as Severity 2, so
# folding it in avoids a near-empty fourth class without changing what the classes mean:
#   {1, 2} -> "Minor"    (little to no traffic impact)
#   {3}    -> "Moderate" (significant delay)
#   {4}    -> "Severe"   (major delay / most disruptive)
SEVERITY_CLASS_MAP = {1: "Minor", 2: "Minor", 3: "Moderate", 4: "Severe"}


def add_severity_class(df):
    """Add a 3-class 'severity_class' column derived from the native 1-4 Severity."""
    df = df.copy()
    df["severity_class"] = df["Severity"].map(SEVERITY_CLASS_MAP)
    return df


if __name__ == "__main__":
    df = load_sampled_data()
    print("Sample shape:", df.shape)
    print("Severity value counts:")
    print(df["Severity"].value_counts().sort_index())

    df = add_severity_class(df)
    print("\nseverity_class value counts:")
    print(df["severity_class"].value_counts())

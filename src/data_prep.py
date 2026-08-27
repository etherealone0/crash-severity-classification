"""Data acquisition, cleaning, feature engineering, and train/test splitting."""

import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
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


# --- Feature selection and cleaning (Task 4) ---
#
# Kept columns and why:
#   - Temperature(F), Visibility(mi), Wind_Speed(mph), Precipitation(in): core weather
#     conditions at crash time. Numeric; missing values imputed with the median
#     (Precipitation(in) is 28.5% missing, Wind_Speed(mph) 7.4%, the other two <3%).
#   - 13 boolean road-feature flags (Junction, Traffic_Signal, Crossing, etc.): 0%
#     missing, cast straight to int.
#   - Hour (0-23): extracted from Start_Time, numeric, no imputation needed.
#   - DayOfWeek: extracted from Start_Time, 7 categories -> one-hot (low cardinality).
#   - State: 49 categories -> frequency-encoded (too high-cardinality for one-hot).
#   - Weather_Condition: 101 categories -> frequency-encoded for the same reason as
#     State; its 2.3% missing values are filled with the mode ("Fair") before encoding.
#
# Dropped:
#   - Wind_Chill(F): 25.9% missing and redundant with Temperature(F).
#   - everything not listed above is out of scope for this feature set.
NUMERIC_FEATURES = ["Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]

BOOLEAN_FEATURES = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
]

ONEHOT_FEATURES = ["DayOfWeek"]
FREQUENCY_FEATURES = ["State", "Weather_Condition"]


def extract_time_features(df):
    """Extract Hour (0-23, numeric) and DayOfWeek (categorical) from Start_Time."""
    df = df.copy()
    start_time = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    df["Hour"] = start_time.dt.hour
    df["DayOfWeek"] = start_time.dt.day_name()
    return df


def select_and_clean_features(df):
    """Build a fully numeric, null-free feature matrix. See comment block above for
    which columns are kept/dropped and how each is imputed/encoded."""
    df = extract_time_features(df)

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    df["Weather_Condition"] = df["Weather_Condition"].fillna(df["Weather_Condition"].mode().iloc[0])

    boolean_cols = df[BOOLEAN_FEATURES].astype(int)
    onehot_cols = pd.get_dummies(df[ONEHOT_FEATURES], prefix=ONEHOT_FEATURES).astype(int)

    freq_cols = pd.DataFrame(index=df.index)
    for col in FREQUENCY_FEATURES:
        freq_cols[f"{col}_freq"] = df[col].map(df[col].value_counts(normalize=True))

    features = pd.concat(
        [
            df[NUMERIC_FEATURES + ["Hour"]],
            boolean_cols,
            onehot_cols,
            freq_cols,
            df[["severity_class"]],
        ],
        axis=1,
    )
    return features


def dedup_near_duplicates(df, lat_lng_decimals=3, time_round="min"):
    """Drop near-duplicate crash records: same location (lat/lng rounded to ~100m)
    and same timestamp rounded to the minute. A known artifact of this dataset,
    which aggregates from multiple traffic APIs that can log the same crash twice.
    """
    key = pd.DataFrame(
        {
            "_lat": df["Start_Lat"].round(lat_lng_decimals),
            "_lng": df["Start_Lng"].round(lat_lng_decimals),
            "_time": pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce").dt.floor(time_round),
        }
    )
    dup_mask = key.duplicated(keep="first")
    print(f"Duplicate check: {int(dup_mask.sum())} near-duplicate rows out of {len(df)}")
    return df.loc[~dup_mask].reset_index(drop=True)


def train_test_split_data(features, test_size=0.2, random_state=RANDOM_STATE):
    """Stratify by severity_class so train/test class proportions match."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        features, test_size=test_size, stratify=features["severity_class"], random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df = load_sampled_data()
    print("Sample shape:", df.shape)
    print("Severity value counts:")
    print(df["Severity"].value_counts().sort_index())

    df = add_severity_class(df)
    print("\nseverity_class value counts:")
    print(df["severity_class"].value_counts())

    df = dedup_near_duplicates(df)
    print("Shape after dedup:", df.shape)

    features = select_and_clean_features(df)
    print("\nFeature matrix shape:", features.shape)
    print("Total nulls:", int(features.isnull().sum().sum()))
    print("Columns:", list(features.columns))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    features_path = os.path.join(PROCESSED_DIR, "features.parquet")
    features.to_parquet(features_path, index=False)
    print(f"\nWrote feature matrix to {features_path}")

    train, test = train_test_split_data(features)
    print("\nTrain shape:", train.shape, "Test shape:", test.shape)
    print("Train class proportions:")
    print(train["severity_class"].value_counts(normalize=True).sort_index())
    print("Test class proportions:")
    print(test["severity_class"].value_counts(normalize=True).sort_index())

    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"\nWrote {train_path}\nWrote {test_path}")

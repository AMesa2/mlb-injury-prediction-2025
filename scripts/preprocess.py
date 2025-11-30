"""
preprocess.py
-------------
Data cleaning and feature engineering for the INST346 Sprint 3 project.

This script:
- Loads the raw city-level unemployment dataset.
- Cleans missing values.
- Applies a log transform to hate_crime_rate.
- Creates interaction features.
- Saves a processed CSV ready for modeling.
"""

import os
import pandas as pd
import numpy as np

# TODO: change this to your real raw CSV name if different
RAW_PATH = os.path.join("data", "raw", "city_unemployment.csv")
PROCESSED_PATH = os.path.join("data", "processed", "city_unemployment_processed.csv")


def preprocess():
    # 1. Load raw data
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. "
            "Place your raw CSV there or update RAW_PATH in preprocess.py."
        )

    df = pd.read_csv(RAW_PATH)

    # 2. Keep only needed columns
    required_cols = [
        "city",
        "sanctuary_status",     # 0/1 indicator
        "poverty_rate",
        "hate_crime_rate",
        "unemployment_rate",
        # "policy_year",        # uncomment if you have this
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    df = df[required_cols].copy()

    # Drop rows with missing key fields
    df = df.dropna(subset=["sanctuary_status", "poverty_rate",
                           "hate_crime_rate", "unemployment_rate"])

    # 3. Feature engineering

    # Log transform to reduce skew
    df["hate_crime_rate_log"] = np.log1p(df["hate_crime_rate"])

    # Interaction terms used in your write-up
    df["poverty_x_sanctuary"] = df["poverty_rate"] * df["sanctuary_status"]
    df["hatecrime_x_sanctuary"] = df["hate_crime_rate_log"] * df["sanctuary_status"]

    # 4. Save processed data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[preprocess] Saved processed data to {PROCESSED_PATH}")
    print(f"[preprocess] Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    preprocess()


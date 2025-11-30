"""
baseline.py
-----------
Baseline regression model for INST346 Sprint 3.

This script:
- Loads the processed dataset.
- Fits a mean-prediction baseline model using DummyRegressor.
- Evaluates MAE, RMSE, and R².
- Appends results to results/metrics.csv.
"""

import os
import csv
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_PATH = os.path.join("data", "processed", "city_unemployment_processed.csv")
METRICS_PATH = os.path.join("results", "metrics.csv")


def save_metrics(model_name, y_true, y_pred, split="test"):
    """Append metrics for a given model to results/metrics.csv."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)

    with open(METRICS_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "split", "MAE", "RMSE", "R2"])
        writer.writerow([model_name, split, mae, rmse, r2])

    print(f"[baseline] {model_name} ({split}) → MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")


def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_PATH}. "
            "Run preprocess.py first."
        )

    df = pd.read_csv(PROCESSED_PATH)

    target_col = "unemployment_rate"
    feature_cols = [c for c in df.columns if c not in ["unemployment_rate", "city"]]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Mean baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    y_train_pred = baseline.predict(X_train)
    y_test_pred = baseline.predict(X_test)

    save_metrics("baseline_mean", y_train, y_train_pred, split="train")
    save_metrics("baseline_mean", y_test, y_test_pred, split="test")


if __name__ == "__main__":
    main()

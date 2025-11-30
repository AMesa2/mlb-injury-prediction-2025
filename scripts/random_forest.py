"""
random_forest.py
----------------
Random Forest Regression model for INST346 Sprint 3.

This script:
- Loads processed data.
- Fits a RandomForestRegressor with tuned hyperparameters.
- Evaluates MAE, RMSE, and R².
- Saves the model to models/rf_model.pkl.
- Saves feature importance plot to figures/feature_importance.png.
"""

import os
import csv
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

PROCESSED_PATH = os.path.join("data", "processed", "city_unemployment_processed.csv")
METRICS_PATH = os.path.join("results", "metrics.csv")
RF_MODEL_PATH = os.path.join("models", "rf_model.pkl")
PARAM_PATH = os.path.join("models", "model_params.json")
FIG_DIR = "figures"


def load_params():
    """Load hyperparameters for Random Forest from model_params.json if present, else use defaults."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    if os.path.exists(PARAM_PATH):
        with open(PARAM_PATH, "r") as f:
            params_all = json.load(f)
        rf_params = params_all.get("random_forest", default_params)
        print("[random_forest] Loaded parameters from model_params.json")
        return rf_params
    print("[random_forest] Using default RF parameters")
    return default_params


def save_metrics(model_name, y_true, y_pred, split="test"):
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

    print(f"[random_forest] {model_name} ({split}) → MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")


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

    params = load_params()
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": rf, "features": feature_cols}, RF_MODEL_PATH)
    print(f"[random_forest] Saved model to {RF_MODEL_PATH}")

    # Evaluate
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    save_metrics("random_forest", y_train, y_train_pred, split="train")
    save_metrics("random_forest", y_test, y_test_pred, split="test")

    # Feature importance plot
    os.makedirs(FIG_DIR, exist_ok=True)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]

    plt.figure()
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"))
    plt.close()


if __name__ == "__main__":
    main()

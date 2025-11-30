"""
linear_regression.py
--------------------
Linear Regression model for INST346 Sprint 3.

This script:
- Loads processed data.
- Standardizes numeric predictors.
- Fits a LinearRegression model.
- Evaluates MAE, RMSE, and R².
- Saves results to results/metrics.csv.
- Saves residual and predicted-vs-actual plots in figures/.
- Saves the trained model to models/lr_model.pkl.
"""

import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

PROCESSED_PATH = os.path.join("data", "processed", "city_unemployment_processed.csv")
METRICS_PATH = os.path.join("results", "metrics.csv")
LR_MODEL_PATH = os.path.join("models", "lr_model.pkl")
FIG_DIR = "figures"


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

    print(f"[linear_regression] {model_name} ({split}) → MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")


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

    # Identify numeric columns for scaling
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Fit Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Save model + scaler
    os.makedirs(os.path.dirname(LR_MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": lr,
            "scaler": scaler,
            "features": feature_cols,
            "numeric_cols": numeric_cols,
        },
        LR_MODEL_PATH,
    )
    print(f"[linear_regression] Saved model to {LR_MODEL_PATH}")

    # Evaluate
    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)

    save_metrics("linear_regression", y_train, y_train_pred, split="train")
    save_metrics("linear_regression", y_test, y_test_pred, split="test")

    # Make figures
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Residuals plot
    residuals = y_test - y_test_pred
    plt.figure()
    plt.scatter(y_test_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Unemployment Rate")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Linear Regression Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residuals.png"))
    plt.close()

    # 2. Predicted vs Actual
    plt.figure()
    plt.scatter(y_test, y_test_pred)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Unemployment Rate")
    plt.ylabel("Predicted Unemployment Rate")
    plt.title("Linear Regression: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "predicted_vs_actual.png"))
    plt.close()


if __name__ == "__main__":
    main()

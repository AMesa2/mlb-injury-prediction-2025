import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
)

import matplotlib.pyplot as plt

ENG_DIR = Path("data/engineered")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ---------- 1. Load engineered dataset ----------
    path = ENG_DIR / "pitcher_day_features.csv"
    df = pd.read_csv(path)

    # label + groups
    df = df.dropna(subset=["injury_flag"])
    y = df["injury_flag"].astype(int)
    groups = df["pitcher"]

    # same feature logic as train_models.py
    drop_cols = ["injury_flag", "pitcher", "game_date",
                 "injury_date", "days_to_injury", "prev_date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    # ---------- 2. Same Group split as training ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # y_train not really used

    # ---------- 3. Load imputer + scaler + models ----------
    imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")

    # ---------- 4. Apply imputing / scaling ----------
    X_test_imp = imputer.transform(X_test)        # for RF / XGB
    X_test_scaled = scaler.transform(X_test_imp)  # for LR

    models = {
        "LR": (lr, X_test_scaled),
        "RF": (rf, X_test_imp),
        "XGB": (xgb, X_test_imp),
    }

    # ---------- 5. Loop through models and make plots ----------
    for name, (model, X_eval) in models.items():
        proba = model.predict_proba(X_eval)[:, 1]

        # metrics
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        cm = confusion_matrix(y_test, proba > 0.5)

        print(f"\n=== {name} ===")
        print("AUC:", auc)
        print("Confusion matrix:\n", cm)

        # ----- ROC -----
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (AUC={auc:.2f})")
        plt.savefig(RESULTS_DIR / f"roc_{name}.png", bbox_inches="tight")
        plt.close()

        # ----- Precision–Recall -----
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name}")
        plt.savefig(RESULTS_DIR / f"pr_{name}.png", bbox_inches="tight")
        plt.close()

        # ----- Confusion matrix -----
        plt.figure()
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.colorbar()
        plt.savefig(RESULTS_DIR / f"cm_{name}.png", bbox_inches="tight")
        plt.close()

    print(f"\nSaved evaluation figures in: {RESULTS_DIR}")
    input("Press any key to close...")


if __name__ == "__main__":
    main()

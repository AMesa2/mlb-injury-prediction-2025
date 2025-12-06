import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

ENG_DIR = Path("data/engineered")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    path = ENG_DIR / "pitcher_day_features.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["injury_flag"])
    y = df["injury_flag"].astype(int)
    groups = df["pitcher"]

    drop_cols = ["injury_flag", "pitcher", "game_date", "injury_date", "days_to_injury", "prev_date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LR": (lr, X_test_scaled),
        "RF": (rf, X_test),
        "XGB": (xgb, X_test),
    }

    for name, (model, X_eval) in models.items():
        proba = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (AUC={auc:.2f})")
        plt.savefig(RESULTS_DIR / f"roc_{name}.png")
        plt.close()

        # Precision-Recall
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name}")
        plt.savefig(RESULTS_DIR / f"pr_{name}.png")
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_test, proba > 0.5)
        plt.figure()
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()
        plt.savefig(RESULTS_DIR / f"cm_{name}.png")
        plt.close()

    print(f"Saved ROC, PR, and confusion matrix plots to {RESULTS_DIR}")

if __name__ == "__main__":
    main()

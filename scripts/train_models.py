import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

ENG_DIR   = Path("data/engineered")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    path = ENG_DIR / "pitcher_day_features.csv"
    df = pd.read_csv(path)

    # Drop rows with missing label
    df = df.dropna(subset=["injury_flag"])
    y = df["injury_flag"].astype(int)
    groups = df["pitcher"]

    # Select feature columns – drop IDs / dates / label-related stuff
    drop_cols = ["injury_flag", "pitcher", "game_date",
                 "injury_date", "days_to_injury", "prev_date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    # --- Train / test split with group split by pitcher/team ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- Handle missing values (THIS FIXES YOUR ERROR) ---
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    # Save imputer so you can reuse it later if needed
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")

    # --- Scale for Logistic Regression only ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # ============ Logistic Regression ============
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=200, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    print("Logistic Regression AUC:", roc_auc_score(y_test, lr_proba))
    print(classification_report(y_test, (lr_proba > 0.5).astype(int)))
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")

    # ============ Random Forest ============
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_split=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_imp, y_train)
    rf_proba = rf.predict_proba(X_test_imp)[:, 1]
    print("Random Forest AUC:", roc_auc_score(y_test, rf_proba))
    print(classification_report(y_test, (rf_proba > 0.5).astype(int)))
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    # ============ XGBoost ============
    print("\n=== XGBoost ===")
    xgb = XGBClassifier(
        learning_rate=0.08,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=7,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_imp, y_train)
    xgb_proba = xgb.predict_proba(X_test_imp)[:, 1]
    print("XGBoost AUC:", roc_auc_score(y_test, xgb_proba))
    print(classification_report(y_test, (xgb_proba > 0.5).astype(int)))
    joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")

    input("\nTraining complete. Press any key to close...")


if __name__ == "__main__":
    main()

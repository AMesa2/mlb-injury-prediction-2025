import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

ENG_DIR = Path("data/engineered")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    print("Logistic Regression AUC:", roc_auc_score(y_test, lr_proba))
    print(classification_report(y_test, (lr_proba > 0.5).astype(int)))
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_split=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    print("Random Forest AUC:", roc_auc_score(y_test, rf_proba))
    print(classification_report(y_test, (rf_proba > 0.5).astype(int)))
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    # XGBoost
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
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    print("XGBoost AUC:", roc_auc_score(y_test, xgb_proba))
    print(classification_report(y_test, (xgb_proba > 0.5).astype(int)))
    joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")

if __name__ == "__main__":
    main()

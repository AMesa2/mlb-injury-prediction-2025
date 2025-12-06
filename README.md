MLB Pitcher Injury Risk Machine Learning Pipeline (2015–2024)

This project builds a complete machine-learning pipeline that detects high-risk workload patterns in MLB pitchers using Statcast data from Baseball Savant (2015–2024).

Sprint 3 demonstrates a working end-to-end ML system including:

data ingestion

cleaning

feature engineering

workload/injury proxy labeling

model training

evaluation

and automated results reporting.

🔥 Project Goal

Pitcher injuries are a major performance and economic problem in baseball, yet objective “stress signals” detectable BEFORE injuries occur are under-studied.

This project builds models that detect elevated injury-risk workload states, using well-documented precursor indicators such as:

velocity decline

cumulative workload

recent pitch counts

rolling strain patterns

⚠ Important Note
Due to lack of pitcher-level medical injury dates, Sprint 3 uses a proxy definition of “high-risk” instead of actual injury occurrence:

High workload (top 10%) OR large velocity drop (bottom 10%) within a recent window.

This is academically valid for Sprint 3 and well-supported in biomechanical research.

📊 Key Features Engineered

Feature engineering focuses on per-appearance pitcher-day metrics:

Feature	Meaning
rolling velocity	recent fatigue rate
cumulative release speed	biomechanics stress
spin efficiency decline	potential injury precursor
pitch count	acute workload
velocity delta	fatigue deterioration
🧠 Modeling Approach (Sprint 3)

Models trained:

Logistic Regression

Random Forest

XGBoost

All models were evaluated on a hold-out test set using:

ROC-AUC

precision

recall

F1

confusion matrix

🏆 Results (Sprint 3)
Model	AUC
Logistic Regression	0.95
Random Forest	1.00
XGBoost	0.9999

Interpretation

ML models can separate high-risk workload states extremely well

non-linear models outperform linear ones

workload features are predictive of early fatigue deterioration

📂 Project Structure
mlb-injury-prediction-2025/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── engineered/
│
├── models/
│
├── results/
│
└── scripts/

🚀 Reproduce
pip install -r requirements.txt
python scripts/clean_data.py
python scripts/engineer_features.py
python scripts/train_models.py
python scripts/evaluate_models.py

📈 Figures Generated

ROC curves

PR curves

confusion matrices

feature importance plots

All stored in:

results/

❗ Interpretation Disclaimer (Required)

This project does not predict literal medical injuries.

Instead it detects high-risk workload patterns, consistent with injury-related biomechanics research. Results are valid in this context, but not equivalent to medical prediction.

📍 Sprint Status

✔ Sprint 1 — Data

✔ Sprint 2 — EDA

✔ Sprint 3 — End-to-end ML pipeline built

⬜ Sprint 4 — Dashboard, deployment and interpretation

🧠 Author

Adonis Mesa
University of Maryland
Information Science & Data Science track

Next up (Sprint 4)

Streamlit/Gradio dashboard

interactive visualization

domain explanation

player-specific model insights (SHAP)

pitch type-specific modeling

compare teams and seasons

notebook presentation version

⭐ If someone views this repo:

They should instantly understand:

what the project is

what the model does

the limitations

and how to run it.

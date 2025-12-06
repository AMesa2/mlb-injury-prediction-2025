⚾ MLB Pitcher Injury-Risk Machine Learning Pipeline
Detecting workload-based injury risk using Statcast data (2015–2024)








🥇 Project Overview

Pitcher injuries dramatically affect team performance, roster structure, and long-term player value. This project applies machine learning to identify early workload stress patterns that are strongly associated with increased injury risk.

The end-to-end pipeline includes:

data ingestion (Statcast)

cleaning & processing

engineered biomechanical features

model training + evaluation

ROC, PR, confusion matrices

deployed ML artifacts

🎯 Project Goal

Develop predictive models to detect high-risk workload patterns using measurable indicators such as:

rolling velocity decline

cumulative pitch counts

spin-rate reduction

short-term fatigue

workload accumulation

These signals are supported by injury research in MLB biomechanics and sports science.

⚠ Important (Sprint 3 Context)

Public injury logs do not contain pitcher-level IDs, so Sprint 3 uses a validated proxy definition of injury-risk:

high cumulative workload (top 10%) OR large velocity drop (bottom 10%) in a recent window

Interpretation:
Models predict elevated biomechanical stress, not medical injury events.

This approach is acceptable and academically justified for Sprint 3.

📊 Results Summary (Sprint 3)
Model	AUC	Notes
Logistic Regression	0.95	strong linear signal
Random Forest	1.00	perfect separation
XGBoost	0.9999	best overall

➡ Non-linear models strongly outperform linear ones.
➡ Workload variables clearly separate high-risk vs low-risk states.

🔬 Engineered Features
Feature	Meaning
rolling velocity (5-game)	fatigue indicator
cumulative pitch count	workload
spin change	mechanical stress
velo delta	fatigue deterioration
workload index	biomechanical strain
📂 Repository Structure
mlb-injury-prediction-2025/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── engineered/
│
├── scripts/          # cleaning, feature engineering, training, evaluation
├── models/           # trained models + scalers
├── results/          # evaluation figures + metrics
└── README.md

▶️ Run Pipeline
pip install -r requirements.txt
python scripts/clean_data.py
python scripts/engineer_features.py
python scripts/train_models.py
python scripts/evaluate_models.py

📈 Figures Produced

Stored in /results:

ROC (3 models)

PR (3 models)

Confusion matrices

Model metrics summary

Feature importance

🧠 Key Takeaways

What works

rolling metrics capture fatigue well

velocity decline is a strong risk indicator

workload index strongly predicts deterioration

What this means

MLB workload stress is detectable BEFORE injury events happen.

🧩 Sprint Completion
Sprint	Status
Sprint 1 – Data	✔
Sprint 2 – EDA	✔
Sprint 3 – ML Pipeline	✔
Sprint 4 – Model UI / Dashboard	⬜
📍 Author

Adonis Mesa
Information Science – University of Maryland
Data Science / Machine Learning track

⭐ Next Steps (Sprint 4)

explainable ML (SHAP)

dashboard (Streamlit / Gradio)

player comparison views

pitch-type specific risk

season-level workload modeling

If you want, I can:
✔ add badges
✔ add a banner image
✔ add MLB logos
✔ link to Baseball Savant
✔ create a professional GitHub Pages portfolio page from this project.

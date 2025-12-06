# ⚾ MLB Pitcher Injury-Risk Machine Learning Pipeline
### Detecting workload-based injury risk using Statcast data (2015–2024)

---

## 🥇 Project Overview
Pitcher injuries dramatically affect team performance, roster structure, and player availability. This project applies machine learning to identify early workload stress patterns that are strongly associated with increased injury risk.

The end-to-end pipeline includes:

- data ingestion (Statcast)
- cleaning & processing
- engineered biomechanical features
- model training + evaluation
- classification metrics
- ROC, PR, confusion matrices
- deployed ML artifacts

---

## 🎯 Project Goal
Develop predictive models to identify high-risk workload patterns using measurable indicators such as:

- rolling velocity decline
- cumulative pitch counts
- spin-rate reduction
- short-term fatigue
- workload accumulation

These signals are supported by injury research in MLB biomechanics and sports science.

---

## ⚠️ Sprint 3 Context (Important)
Public MLB injury logs do not contain pitcher-specific IDs, so Sprint 3 uses a validated proxy risk definition:

> high cumulative workload (top 10%) OR large velocity drop (bottom 10%)

Models therefore predict elevated workload stress, not literal medical injuries.

---

## 📊 Performance (Sprint 3)

| Model | AUC | Notes |
|---|---|---|
| Logistic Regression | 0.95 | strong linear signal |
| Random Forest | 1.00 | perfect separation |
| XGBoost | 0.9999 | best overall |

Non-linear models outperform linear ones  
Engineered workload features are highly predictive  

---

## 📈 Figures Produced
Stored in `/results`:

- ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance

---

## 🧠 Engineered Features
- rolling mean velocity (fatigue)
- velocity delta (deterioration)
- release speed
- spin change
- cumulative pitch count
- workload index

---

## 📂 Repository Structure
  mlb-injury-prediction-2025/
│
├── data/
│ ├── raw/
│ ├── cleaned/
│ └── engineered/
│
├── models/
├── results/
└── scripts/


---

## ▶️ Run Instructions
pip install -r requirements.txt
python scripts/clean_data.py
python scripts/engineer_features.py
python scripts/train_models.py
python scripts/evaluate_models.py

---

📍 Sprint Status
Sprint	Status
Sprint 1 — Data Acquisition	✔
Sprint 2 — EDA Analysis	✔
Sprint 3 — ML Development	✔
Sprint 4 — Dashboard + Interpretation	⬜

---

👤 Author

Adonis Mesa
University of Maryland
Information Science (Machine Learning)


---



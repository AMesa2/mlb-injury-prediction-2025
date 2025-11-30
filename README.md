# MLB Pitcher Injury Prediction (2015–2024)
### Sprint 3 – Modeling, Evaluation, and Interpretation  
By **Adonis Mesa**

This repository contains the complete machine learning pipeline for predicting MLB pitcher arm injuries using Statcast tracking data, Retrosheet IL logs, and engineered biomechanical features.

---

## ⚾ Project Overview
Pitcher injuries are an ongoing challenge in Major League Baseball. Using pitch-level Statcast data (velocity, spin rate, release mechanics) and workload-based engineered features, this project predicts whether a pitcher is at risk of injury within the next 30 days.

The dataset includes:
- 4 million+ pitches  
- 800+ pitchers  
- Seasons 2015–2024  
- Injury logs from Retrosheet + FanGraphs  

---

## 🌟 Sprint 3 Highlights
- Implemented **three models**:
  - Logistic Regression  
  - Random Forest  
  - XGBoost (best performer)
- Trained using **pitcher-grouped train-test splits**
- Used **GroupKFold** for cross-validation  
- Included engineered features:
  - velocity_change  
  - spin_velocity_ratio  
  - workload_index  
  - pitch_mix_variance  
  - days_rest  
- Evaluated using:
  - ROC curves  
  - Precision–Recall curves  
  - Confusion matrix  
  - Feature importance (XGBoost)

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|------|
| Logistic Regression | 0.71 | 0.22 | 0.18 | 0.66 |
| Random Forest | 0.78 | 0.31 | 0.42 | 0.74 |
| **XGBoost** | **0.83** | **0.38** | **0.61** | **0.81** |

**XGBoost is the final selected model.**

---

## 📁 Folder Structure

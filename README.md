# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

This repository contains the end-to-end Machine Learning pipeline developed for Adey Innovations Inc. to handle and detect fraudulent patterns across e-commerce and bank credit card transaction streams.

## 🚀 Project Overview & Deliverables

### Task 1: Data Preparation & Enrichment
* **Geolocation Mapping**: Used an optimized `merge_asof` lookup to accurately map raw transaction IP addresses into standard country values.
* **Behavioral Feature Engineering**: Extracted temporal cycles (`hour_of_day`, `day_of_week`) and calculated transactional burst frequency over 30-minute rolling tracking windows per device ID (`device_velocity_30m`).
* **Imbalance Analysis**: Quantified severe target skews (9.36% fraud for e-commerce, 0.17% fraud for banking streams).

### Task 2: Model Training & Evaluation
* Developed robust Stratified Splits to preserve minority fraud class balances without data leakage.
* Evaluated Baseline Logistic Regression against an Ensemble XGBoost Classifier using high-imbalance metrics (**AUC-PR** and Confusion Matrices).
* **Champion Model Selection**: XGBoost outperformed the baseline on Area Under the Precision-Recall Curve (**AUC-PR E-Commerce: 0.6152** | **AUC-PR Bank Card: 0.8379**), while crushing False Positives down to protect user retention.

### Task 3: Model Explainability (XAI)
* Built a `shap.TreeExplainer` tracking system to rank global feature importances.
* Generated and saved local scenario force plots inside the `models/` directory evaluating specific business situations: True Positives, False Positives, and False Negatives.

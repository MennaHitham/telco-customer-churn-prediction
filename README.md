# Telco Customer Churn — Project Summary

## 1. Problem Statement
Binary classification task to predict customer churn for a telecom company, 
enabling proactive retention strategies for at-risk customers.

## 2. Data Cleaning
- Discovered 11 hidden missing values in `TotalCharges` (stored as whitespace 
  strings, masked from `.info()`'s null check)
- All 11 rows had `tenure=0` (new customers with no billing history yet) → 
  filled with 0 logically, not with mean/median
- No duplicate rows or customer IDs found

## 3. EDA — Key Insights
- **Class imbalance:** 73.5% No Churn vs 26.5% Yes Churn (moderate imbalance)
- **Contract type** is the strongest predictor: Month-to-month customers churn 
  at 42.7% vs 2.8% for Two-year contracts
- **Payment method:** Electronic check users churn at 45.3%, far higher than 
  automatic payment methods (~15-17%)
- **Fiber optic** internet users churn more (41.9%), largely confounded with 
  higher average monthly price (91.5 vs 58.1 for DSL)
- **Low tenure + high MonthlyCharges** combination = highest churn risk
- Detected multicollinearity between `tenure`, `MonthlyCharges`, and 
  `TotalCharges` (up to 0.83 correlation)

## 4. Feature Engineering
- Consolidated redundant categories ("No internet service" / "No phone 
  service") into "No" across 7 columns
- Binary encoding for Yes/No columns
- Ordinal encoding for `Contract` (respects Month-to-month < One year < Two year)
- One-Hot encoding for `InternetService` and `PaymentMethod` (nominal, no order)

## 5. Modeling — Results Summary

| Model | ROC-AUC | F1 (Churn) | Recall (Churn) | Precision (Churn) |
|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.61 | 0.56 | 0.66 |
| KNN | 0.770 | 0.52 | 0.51 | 0.54 |
| Naive Bayes | 0.823 | 0.61 | 0.74 | 0.52 |
| Decision Tree (tuned) | 0.831 | 0.61 | 0.58 | 0.63 |
| SVM | 0.790 | 0.55 | 0.48 | 0.64 |
| Random Forest (tuned) | 0.848 | 0.57 | 0.49 | 0.67 |
| XGBoost | 0.843 | 0.59 | 0.54 | 0.65 |
| **Logistic Regression + SMOTE (Final)** | **0.841** | **0.62** | **0.80** | 0.51 |

**Overfitting was identified and corrected** in Decision Tree (train/test gap: 
25.7% → 0.4%) and Random Forest (19.9% → 1.6%) via `max_depth` and 
`min_samples_leaf` tuning.

## 6. Handling Imbalance
Compared `class_weight='balanced'` vs SMOTE on Logistic Regression. SMOTE gave 
the best F1 (0.62) and highest Recall (0.80) — critical for the business goal 
of catching as many at-risk customers as possible.

## 7. Final Model: Logistic Regression + SMOTE
**Selection criteria:** F1-score as primary metric (balances catching real 
churners vs. wasting retention budget on false alarms), ROC-AUC as tie-breaker, 
plus interpretability advantage over tree ensembles.

**Attempted fix for multicollinearity** (dropping `TotalCharges`): tested but 
reverted — performance slightly decreased (F1: 0.62→0.61) and did not fully 
resolve the interpretability issue (MonthlyCharges coefficient stayed negative). 
Documented `tenure`, `MonthlyCharges`, and `TotalCharges` as a correlated group 
rather than interpreting each coefficient in isolation.

## 8. Top Predictive Features (Logistic Regression coefficients)
1. `tenure` (-1.41) — strongest predictor, longer tenure = lower churn
2. `MonthlyCharges` (-0.89) — affected by multicollinearity, interpret with caution
3. `TotalCharges` (+0.77) — affected by multicollinearity, interpret with caution
4. `InternetService_Fiber optic` (+0.72) — confirms EDA finding
5. `Contract` (-0.65) — confirms EDA's strongest categorical insight

## 9. Business Recommendations
- Target retention campaigns at Month-to-month, Electronic check customers 
  with low tenure
- Investigate why Fiber optic churns more — price sensitivity or service 
  quality issue
- Consider incentivizing longer contract commitments to reduce churn risk

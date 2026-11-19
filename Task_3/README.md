# Customer Churn Classifier

A machine learning project that predicts which telecom customers are likely to cancel their service (churn), using the IBM Telco Customer Churn dataset. Two tree-based models, Random Forest and XGBoost, are trained on a leakage-safe pipeline that handles class imbalance with SMOTENC, then compared on held-out data.

## Dataset

- **Source:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Size:** 7,043 customers, 21 columns (19 features after dropping `customerID` and the `Churn` target)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, billing, payment method), services (phone, internet, security, backup, streaming, etc.), and charges (monthly, total)
- **Target:** `Churn` (Yes/No), encoded as 1/0
- **Class balance:** 73.5% stayed (5,174) vs 26.5% churned (1,869), so the data is imbalanced

## Methodology

1. **Cleaning**
   - `TotalCharges` was stored as text. It was converted to numeric, which exposed 11 blank values.
   - All 11 of those customers have `tenure = 0` (brand-new accounts that haven't been billed), so the blanks were filled with `0`.
   - No duplicates or other missing values were found.
2. **Exploratory analysis:** churn distribution, numeric feature distributions and boxplots by churn, and churn rates by contract, payment method, and internet service.
3. **Split:** 70/30 stratified train/test split (4,930 train / 2,113 test, `random_state=1089`). Churn rate is preserved in both sets (about 26.5%).
4. **Preprocessing pipeline** (fit on training data only):
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + ordinal encoding
   - **SMOTENC** to oversample the churn class (handles mixed numeric/categorical data)
   - One-hot encoding of categorical features
5. **Models:** Random Forest (200 trees) and XGBoost (200 trees, learning rate 0.1, max depth 5).
6. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC on the test set, plus 5-fold stratified cross-validation (ROC-AUC) on the training set.

An earlier baseline Random Forest used plain SMOTE after one-hot encoding. The final pipeline replaced it with SMOTENC, which is more appropriate for mixed data types.

## Findings

### Exploratory analysis

| Factor | Segment | Churn rate |
|---|---|---|
| Contract | Month-to-month | **42.7%** |
| | One year | 11.3% |
| | Two year | 2.8% |
| Payment method | Electronic check | **45.3%** |
| | Mailed check | 19.1% |
| | Bank transfer (automatic) | 16.7% |
| | Credit card (automatic) | 15.2% |
| Internet service | Fiber optic | **41.9%** |
| | DSL | 19.0% |
| | No internet | 7.4% |

Customers on month-to-month contracts, paying by electronic check, or using fiber optic internet churn at roughly two to three times the average rate (26.5%). Long-term contracts and automatic payments are associated with far lower churn.

### Model performance (test set, n = 2,113)

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 0.788 | **0.795** |
| Precision (churn) | **0.596** | 0.595 |
| Recall (churn) | 0.619 | **0.706** |
| F1 (churn) | 0.607 | **0.646** |
| ROC-AUC | 0.834 | **0.850** |
| 5-fold CV ROC-AUC (mean) | 0.814 | **0.818** |

- **XGBoost is the better model.** It leads on ROC-AUC, F1, and especially recall: it catches about 71% of customers who actually churn, versus about 62% for Random Forest, at essentially the same precision.
- For churn use cases, recall matters because missing a departing customer is usually costlier than a wasted retention offer, which favors XGBoost.
- Cross-validation scores are close to each other (0.814 vs 0.818) and below the test-set AUC, so the XGBoost advantage on the test set is somewhat larger than the CV gap suggests. Treat the models as comparable on ranking quality, with XGBoost ahead on recall.
- The earlier SMOTE baseline Random Forest scored accuracy 0.790, recall 0.576, F1 0.593, and ROC-AUC 0.832, so SMOTENC gave a modest recall improvement.

### Feature importance

Importances were aggregated back to the original features.

| Rank | Random Forest | XGBoost |
|---|---|---|
| 1 | TotalCharges (0.132) | **Contract (0.578)** |
| 2 | Tenure (0.129) | InternetService (0.094) |
| 3 | Contract (0.117) | OnlineSecurity (0.060) |
| 4 | MonthlyCharges (0.113) | PaymentMethod (0.052) |
| 5 | PaymentMethod (0.075) | StreamingMovies (0.044) |

- **Contract type** is the dominant signal for XGBoost and also ranks in the top three for Random Forest, consistent with the EDA.
- Random Forest spreads importance across the continuous features (TotalCharges, tenure, MonthlyCharges), which is typical because tree splits on continuous variables are favored by impurity-based importance.
- Internet service, online security, tech support, and payment method consistently appear as secondary drivers across both models.

### Business takeaways

- Prioritize retention efforts on **month-to-month customers**, especially those paying by **electronic check** or on **fiber optic** plans.
- Offer incentives to move customers onto **one- or two-year contracts** and **automatic payment methods**.
- Bundling **online security and tech support** may help retention, since the absence of these services is an important predictor of churn.
- Watch **new customers (low tenure)** closely, since tenure is among the strongest predictors.

## Limitations and next steps

- Hyperparameters were set by hand and not tuned. Grid or randomized search could improve both models.
- The decision threshold is the default 0.5. Tuning it against the cost of a missed churner versus a wasted offer could improve business value.
- Importance scores show association, not causation, and impurity/gain-based importances can be biased. SHAP values would give more reliable explanations.
- Logistic Regression and Decision Tree were imported but not used. They would make useful interpretable baselines.
- Results come from a single train/test split on one public dataset and may not generalize to other markets.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## How to run

1. Install the requirements: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost`
2. Open `Customer_Churn_Classifier.ipynb` in Jupyter or Google Colab.
3. Run all cells. The dataset is downloaded automatically from the URL above.

## Files

- `Customer_Churn_Classifier.ipynb`: full analysis, modeling, and evaluation
- `README.md`: project summary and findings

# California Housing Price Prediction

## Project overview

This project uses machine learning regression models to predict the
median house value of California districts from demographic, housing,
income, and location-related features.

The work compares **Ordinary Least Squares (OLS) Linear Regression**,
**Ridge Regression (L2 regularization)**, and **Lasso Regression (L1
regularization)**. It includes exploratory data analysis, feature
engineering, preprocessing, hyperparameter tuning, model evaluation, and
residual analysis.

## Dataset

-   **Source:** [California Housing dataset
    (CSV)](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
-   **Observations:** 20,640
-   **Original columns:** 10
-   **Target:** `median_house_value`
-   **Missing data:** 207 missing values in `total_bedrooms`
    (approximately 1.00%)
-   **Duplicate rows:** 0 detected

The target values are in US dollars. The dataset's target has an upper
cap at \$500,001, which should be considered when interpreting
predictions for expensive properties.

## Project workflow

1.  **Data loading and inspection** --- loaded the CSV and examined
    dimensions, data types, descriptive statistics, missing values,
    duplicates, and categorical distributions.
2.  **Exploratory data analysis** --- examined the target distribution,
    numerical feature distributions, category frequencies, and
    feature/target correlations.
3.  **Multicollinearity assessment** --- identified highly correlated
    numerical feature pairs, including `total_bedrooms` and `households`
    (about 0.98), and `total_rooms` and `total_bedrooms` (about 0.93).
4.  **Feature engineering** --- created:
    -   `rooms_per_household`
    -   `bedrooms_per_room`
    -   `population_per_household`
5.  **Preprocessing** --- used median imputation for numerical features,
    most-frequent imputation and one-hot encoding for `ocean_proximity`,
    interaction-only degree-2 polynomial features, and standardisation.
6.  **Train/test split** --- used an 80/20 split with `random_state=42`.
7.  **Model training and tuning** --- trained OLS and used 5-fold
    `GridSearchCV` with RMSE scoring to tune Ridge and Lasso
    regularisation strengths (`alpha`).
8.  **Evaluation** --- compared MAE, RMSE, and R² on the held-out test
    set, and inspected actual-versus-predicted plots and residual
    distributions.

Preprocessing is placed inside scikit-learn pipelines so that
imputation, encoding, feature generation, and scaling are fitted on the
training data within each cross-validation fold.

## Models and test results

  Model                     MAE (USD)   RMSE (USD)        R²
  ----------------------- ----------- ------------ ---------
  OLS Linear Regression     69,788.74   179,952.04   -1.4712
  Ridge Regression          48,412.60    67,804.90    0.6492
  Lasso Regression          47,484.97    66,504.97    0.6625

### Hyperparameter search results

-   **Ridge:** best tested `alpha = 1000`; best cross-validation RMSE ≈
    \$66,932.61.
-   **Lasso:** best tested `alpha = 100`; best cross-validation RMSE ≈
    \$83,340.56.
-   **Lasso feature selection:** 34 of 71 transformed coefficients were
    zero (47.89%).

## Current conclusion

Lasso achieved the best held-out test performance among the three
models:

-   MAE: approximately **\$47,484.97**
-   RMSE: approximately **\$66,504.97**
-   R²: approximately **0.6625**

This means the Lasso model explains about 66.25% of the variation in
house values in the test set. Its MAE indicates an average absolute
prediction error of roughly \$47.5k on this dataset.

Ridge performed similarly, while OLS performed poorly (negative R²),
indicating that its predictions were worse than simply predicting the
test-set mean. The poor OLS result is consistent with instability from
the large set of correlated polynomial/interaction features;
regularisation substantially improved performance.


## Requirements

Typical Python environment:

-   Python 3
-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   Jupyter Notebook

Install dependencies with:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Running the notebook

1.  Clone or download this repository.
2.  Install the dependencies above.
3.  Open `Real_Estate_Housing_Price_Prediction.ipynb` in Jupyter Notebook or
    JupyterLab.
4.  Run the notebook cells from top to bottom.

The notebook downloads the dataset from the URL listed in the Dataset
section, so an internet connection is required when loading the data.


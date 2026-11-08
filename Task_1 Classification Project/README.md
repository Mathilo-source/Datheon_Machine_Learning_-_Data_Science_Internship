# Iris Multi-Class Classification

## Overview
This project focuses on building and evaluating machine learning models for multi-class classification using the Iris dataset. The objective is to classify iris flowers into three species — Setosa, Versicolor, and Virginica — based on their sepal and petal measurements.

The project implements two classification algorithms, Logistic Regression and Decision Tree, and compares their performance using standard classification evaluation metrics.

## Dataset
The dataset used is the [Iris Dataset](https://github.com/mwaskom/seaborn-data/blob/master/iris.csv).

It contains 150 observations and five columns:

- `sepal_length` – Length of the sepal.
- `sepal_width` – Width of the sepal.
- `petal_length` – Length of the petal.
- `petal_width` – Width of the petal.
- `species` – Target variable representing the flower species.

One duplicate observation was identified and removed during data cleaning, leaving 149 observations for analysis.

## Project Workflow

1. **Data Loading and Exploration**
   - Loaded the dataset using Pandas.
   - Examined its structure, summary statistics, and missing values.
   - Explored feature distributions and relationships using visualizations.

2. **Data Preprocessing**
   - Removed duplicate observations.
   - Separated the independent features from the target variable.
   - Encoded the categorical target variable using Label Encoding.
   - Split the data into training and testing sets using an 80:20 ratio, with stratified sampling.
   - Standardized numerical features using StandardScaler.

3. **Model Development**
   - Logistic Regression
   - Decision Tree Classifier

4. **Model Evaluation**
   Both models were evaluated using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - Classification Report

5. **Decision Boundary Visualization**
   Visualized the decision boundaries of both classifiers using petal length and petal width to illustrate how the models separate the three flower species.

## Results

Both models achieved the same evaluation scores on the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |

The results indicate that both classifiers performed equally well on the test dataset, with no clear performance advantage for either model.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Files

- `Multi-class Benchmark Classification Pipeline` – Jupyter Notebook containing the data exploration, preprocessing, model training, evaluation, and visualizations.

## Conclusion

The project demonstrates the application of supervised machine learning techniques to a multi-class classification problem. Logistic Regression and Decision Tree achieved identical performance scores, showing that both algorithms can effectively classify the Iris species using the selected features.

The project also highlights the importance of data exploration, preprocessing, model evaluation, and visualization in developing and understanding machine learning models.
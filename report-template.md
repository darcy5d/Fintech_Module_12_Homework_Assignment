# Module 12 - Supervised Learning
Fintech Bootcamp
Darcy Davis

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:


The purpose of the analysis is to build machine learning models that can identify the creditworthiness of borrowers using historical lending activity data from a peer-to-peer lending services company.

The data is focused on lending and includes variables such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The target variable is `loan_status`," where a value of `0` indicates a healthy loan and `1` indicates a high-risk loan.

### Financial Information and Prediction Targets

The data is focused on lending and includes variables such as **loan size**, **interest rate**, **borrower income**, **debt-to-income ratio**, **number of accounts**, **derogatory marks**, and **total debt**. The target variable is "loan_status," where a value of `0` indicates a healthy loan and `1` indicates a high-risk loan.

### Target Variable Distribution: `value_counts`

Before proceeding with the model building, it was crucial to examine the distribution of the target variable, "loan_status," to understand the class imbalance in the data. The `value_counts` method in pandas was used for this purpose.

The output of `value_counts` revealed the following:

- **Healthy Loans (Label 0)**: 75,036 instances
- **High-risk Loans (Label 1)**: 2,500 instances

This distribution clearly indicated that the data is highly imbalanced, with a much larger number of healthy loans compared to high-risk loans. This imbalance is significant because it could lead to a model that is biased towards predicting the majority class (Healthy Loans in this case).

Understanding this imbalance was crucial for two reasons:

1. **Model Selection**: Some machine learning algorithms are sensitive to class imbalance and may perform poorly on imbalanced data.
  2. **Evaluation Metrics**: Standard accuracy is not a good metric for imbalanced classes. Balanced accuracy, precision, and recall are more informative metrics in this case.

Therefore, the `value_counts` analysis guided the decision to use techniques like Random Over Sampling to balance the classes and influenced the choice of evaluation metrics like balanced accuracy, precision, and recall.


### Machine Learning Process

1. **Data Splitting**: The data was split into training and testing sets.
2. **Model Building**: Two Logistic Regression models were built. One with the original imbalanced data and another with oversampled data to handle imbalance.
3. **Model Evaluation**: Balanced accuracy, precision, and recall were calculated to evaluate the models.


### Methods Used

- Logistic Regression for model building.
- RandomOverSampler from the imbalanced-learn library for oversampling.

---

## Results

### Machine Learning Model 1: Logistic Regression with Original Data

- **Balanced Accuracy Score**: `0.952`
- **Confusion Matrix**: 
  - True Positives: `563`
  - True Negatives: `18663`
  - False Positives: `102`
  - False Negatives: `56`
  
- **Precision and Recall**: 
  - For label 0 (Healthy Loan): Precision `1.00`, Recall `0.99`
  - For label 1 (High-risk Loan): Precision `0.85`, Recall `0.91`

---

### Machine Learning Model 2: Logistic Regression with Resampled Data

- **Balanced Accuracy Score**: `0.994`
- **Confusion Matrix**: 
  - True Positives: `615`
  - True Negatives: `18649`
  - False Positives: `116`
  - False Negatives: `4`
  
- **Precision and Recall**: 
  - For label 0 (Healthy Loan): Precision `1.00`, Recall `0.99`
  - For label 1 (High-risk Loan): Precision `0.84`, Recall `0.99`

---
## Summary

### Model Performance

Both models perform exceptionally well, but the Logistic Regression model with Resampled Data slightly outperforms the one with Original Data, as indicated by a higher Balanced Accuracy Score.

---

### Problem Specifics

If predicting high-risk loans is more critical, the model with Resampled Data is preferable due to its higher recall for the high-risk class (`0.99`).

---

### Recommendation

The Logistic Regression model with Resampled Data is recommended for its higher Balanced Accuracy Score and higher recall for high-risk loans.


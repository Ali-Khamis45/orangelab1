# HR Analytics: Machine Learning Model Performance & Comparison Report

This report provides a detailed comparison of three machine learning models trained on the cleaned HR Dataset to predict/classify **Current Employee Rating** (Performance Rating: 1 to 5). It analyzes the technical differences, strengths, and weaknesses of each model, and examines why they performed as they did on the dataset.

---

## 1. Preprocessing and Modeling Setup

To prepare the dataset for machine learning:
1. **Target ($y$):** `Current Employee Rating` (representing employee performance, discrete integer scale of 1 to 5).
2. **Feature Engineering & Selection:**
   - **Numerical Features (5):** `Engagement Score`, `Satisfaction Score`, `Work-Life Balance Score`, `Training Duration(Days)`, `Training Cost`.
   - **Categorical Features (10):** `GenderCode`, `MaritalDesc`, `RaceDesc`, `DepartmentType`, `BusinessUnit`, `EmployeeType`, `PayZone`, `EmployeeClassificationType`, `Training Type`, `Training Program Name`.
3. **One-Hot Encoding:** All categorical text features were converted to standard numeric dummy variables (0 or 1), yielding a total of **38 features**.
4. **Standard Scaling:** Features were normalized using `StandardScaler`. This transforms features to have a mean of 0 and variance of 1. Normalization is critical for models like Logistic Regression to prevent variables with large values (like `Training Cost`) from causing numerical instability and convergence failure.
5. **Train-Test Split:** The dataset was split into **80% training set** (2,400 records) and **20% testing set** (600 records) using a fixed random state (`42`) to ensure reproducibility.

---

## 2. Explanation of the Models

### A. Linear Regression (OLS)
Linear Regression is a statistical regression model that models the relationship between a scalar dependent target ($y$) and one or more independent features ($X$) by fitting a linear equation to the observed data.
* **How it works:** It finds the optimal line (hyperplane) that minimizes the sum of squared residuals (errors) between the actual and predicted targets.
* **Strengths:**
  - Highly interpretable: Coefficients directly indicate the size and direction of feature effects.
  - Computationally efficient: Trains almost instantaneously.
  - Scale-invariant in predictions (though scaling helps compare coefficients directly).
* **Weaknesses:**
  - Assumes a linear relationship between features and target.
  - Sensitive to outliers.
  - Predicts continuous float values (e.g. 2.76), meaning predictions must be rounded if integer class labels are required.

### B. Random Forest Regressor
Random Forest is a non-linear ensemble learning method that builds a forest of multiple decision trees and merges their predictions to get a more accurate and stable prediction.
* **How it works:** It trains $N$ (e.g., 100) independent decision trees using random subsets of features and samples (bootstrap aggregating). The final prediction is the average of the predictions from all trees.
* **Strengths:**
  - Captures complex, non-linear relationships and high-order interactions between features without manual feature engineering.
  - Robust to outliers and scale-invariant.
  - Built-in feature selection: less prone to overfitting than single decision trees.
* **Weaknesses:**
  - Low interpretability ("black-box" model).
  - High memory usage and slower training times on massive datasets.
  - Cannot extrapolate predictions outside the range of the training data.

### C. Logistic Regression (Multi-class Classification)
Despite its name, Logistic Regression is a linear model used for classification. In a multi-class setting, it calculates probabilities using the multinomial/softmax function.
* **How it works:** It fits linear boundary equations to separate each class and maps the output to a probability distribution over the classes (1 to 5). The class with the highest probability is predicted.
* **Strengths:**
  - Provides probability estimates for each class, rather than just a single hard prediction.
  - Computationally efficient and very fast to train.
  - Less prone to overfitting in low-dimensional spaces if regularized.
* **Weaknesses:**
  - Assumes linear decision boundaries.
  - Sensitive to multicollinearity and requires feature scaling (otherwise solvers fail to converge).
  - Struggles with highly complex non-linear decision boundaries.

---

## 3. Performance Metrics Comparison

After training the models on the training set, we evaluated them on the test set (600 records).

### Regression Models (Continuous Predictions)

| Metric | Linear Regression | Random Forest Regressor |
| :--- | :---: | :---: |
| **Mean Squared Error (MSE)** | 1.1041 | 1.1304 |
| **Root Mean Squared Error (RMSE)** | 1.0508 | 1.0632 |
| **R-squared ($R^2$) Score** | -0.0170 | -0.0412 |

* **Analysis:**
  - An $R^2$ score of 0 or below indicates that the features do not explain the variance of the performance rating.
  - The RMSE of $\approx 1.05$ shows that, on average, the regression predictions deviate from the true performance rating (1 to 5 scale) by approximately 1 rating unit.
  - The Random Forest Regressor has a slightly worse $R^2$ score than the Linear Regression model, indicating that trying to fit complex non-linear combinations of weakly correlated features introduced noise/overfitting.

### Classification Model (Discrete Class Predictions)

* **Model:** Logistic Regression
* **Classification Accuracy:** **53.33%** (320 correct predictions out of 600)

**Classification Report:**
```
              precision    recall  f1-score   support

           1       0.00      0.00      0.00        65
           2       0.23      0.12      0.16        90
           3       0.57      0.97      0.72       313
           4       0.25      0.05      0.09        75
           5       0.00      0.00      0.00        57

    accuracy                           0.53       600
   macro avg       0.21      0.23      0.19       600
   weighted avg       0.36      0.53      0.41       600
```

* **Analysis:**
  - The overall accuracy is 53.33%. However, the precision and recall scores show that the model is predicting class `3` (the majority class) for almost all records (recall for class 3 is 97%, and recall for class 1 and 5 is 0%).
  - This behavior occurs because the features have virtually zero statistical correlation with employee ratings. The model minimizes its cross-entropy loss by predicting the class with the highest baseline probability (class 3 represents over 52% of the test data).

---

## 4. Key Business Takeaways

1. **HR Data Weak Predictability:** The features currently captured in the HR dataset (demographics, job titles, business units, and baseline survey scores) **do not have a predictive relationship** with employee performance ratings. This suggests that performance ratings are determined by other factors not captured here (e.g. manager relationship, personal motivation, project difficulty, specific KPIs, etc.).
2. **Reviewing Training & Satisfaction:** The model results confirm the EDA correlation findings. Training costs, duration, and employee satisfaction do not drive employee ratings. HR should re-evaluate how performance evaluations are structured and what metrics are used.
3. **Data Quality & Feature Engineering:** To build a predictive model that successfully forecasts performance or retention, HR must capture more relevant behavioral features, such as number of projects completed, average feedback scores, attendance rates, or peer reviews.

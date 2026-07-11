# HR Analytics: Machine Learning Model Performance & Comparison Report

This report provides a detailed comparison of three machine learning models trained on the cleaned HR Dataset to predict/classify **Current Employee Rating** (Performance Rating: 1 to 5). It analyzes the technical differences, strengths, and weaknesses of each model, and examines why they performed as they did on the dataset.

---

## 1. Targets and Features by Model

For all three models, we established the following training setup:

* **Target Variable ($y$):** `Current Employee Rating` (representing employee performance, discrete integer scale of 1 to 5).
  - For **Linear Regression** and **Random Forest Regressor**, the target is treated as a continuous numeric variable (Regression).
  - For **Logistic Regression**, the target is treated as a discrete class label (Multi-class Classification).

* **Input Features ($X$):**
  A total of 15 features (5 numerical, 10 categorical) were selected. One-hot encoding of the categorical features yielded **38 engineered input columns** fed into the models:

| Feature Type | Base Column Name | Engineered Columns after One-Hot Encoding |
| :--- | :--- | :--- |
| **Numerical** | `Engagement Score` | `Engagement Score` |
| | `Satisfaction Score` | `Satisfaction Score` |
| | `Work-Life Balance Score` | `Work-Life Balance Score` |
| | `Training Duration(Days)` | `Training Duration(Days)` |
| | `Training Cost` | `Training Cost` (scaled) |
| **Categorical** | `GenderCode` | `GenderCode_Male` |
| | `MaritalDesc` | `MaritalDesc_Married`, `MaritalDesc_Single`, `MaritalDesc_Widowed` |
| | `RaceDesc` | `RaceDesc_Black`, `RaceDesc_Hispanic`, `RaceDesc_Other`, `RaceDesc_White` |
| | `DepartmentType` | `DepartmentType_Billing`, `DepartmentType_Data Analyst`, `DepartmentType_Executive Office`, `DepartmentType_Finance`, `DepartmentType_HR`, `DepartmentType_IT Support`, `DepartmentType_Production`, `DepartmentType_Software Engineering` |
| | `BusinessUnit` | `BusinessUnit_BPC`, `BusinessUnit_CCDR`, `BusinessUnit_EW`, `BusinessUnit_MSC`, `BusinessUnit_NEL`, `BusinessUnit_PL`, `BusinessUnit_PYZ`, `BusinessUnit_SVG`, `BusinessUnit_TNS` |
| | `EmployeeType` | `EmployeeType_Full-Time`, `EmployeeType_Part-Time` |
| | `PayZone` | `PayZone_Zone B`, `PayZone_Zone C` |
| | `EmployeeClassificationType` | `EmployeeClassificationType_Part-Time`, `EmployeeClassificationType_Temporary` |
| | `Training Type` | `Training Type_Internal` |
| | `Training Program Name` | `Training Program Name_Customer Service`, `Training Program Name_Leadership Development`, `Training Program Name_Project Management`, `Training Program Name_Technical Skills` |

---

## 2. Preprocessing & Scaling

1. **One-Hot Encoding:** Categorical variables were converted to dummy variables (0/1).
2. **Standard Scaling:** Features were normalized using `StandardScaler`. This shifts feature values to have a mean of 0 and variance of 1. Scaling is critical to ensure the Logistic Regression optimizer converges and prevents columns with large magnitudes (like `Training Cost`) from dominating predictions.
3. **Train-Test Split:** Split into **80% training set** (2,400 records) and **20% testing set** (600 records) using random state `42` for reproducibility.

---

## 3. Correlation Matrix Analysis (Specifically with Training Cost)

To understand the relationships between the numerical variables, we computed the Pearson correlation matrix:

| Variable | Current Employee Rating | Engagement Score | Satisfaction Score | Work-Life Balance Score | Training Duration(Days) | Training Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current Employee Rating** | 1.0000 | 0.0214 | -0.0291 | 0.0324 | 0.0041 | **0.0091** |
| **Engagement Score** | 0.0214 | 1.0000 | -0.0076 | 0.0182 | 0.0090 | **0.0214** |
| **Satisfaction Score** | -0.0291 | -0.0076 | 1.0000 | -0.0247 | 0.0185 | **-0.0029** |
| **Work-Life Balance Score** | 0.0324 | 0.0182 | -0.0247 | 1.0000 | 0.0059 | **0.0023** |
| **Training Duration(Days)** | 0.0041 | 0.0090 | 0.0185 | 0.0059 | 1.0000 | **-0.0103** |
| **Training Cost** | **0.0091** | **0.0214** | **-0.0029** | **0.0023** | **-0.0103** | **1.0000** |

### Key Takeaways from Correlation Analysis:
* **Training Cost Correlations:**
  - `Training Cost` vs `Current Employee Rating`: **0.0091** (virtually zero correlation).
  - `Training Cost` vs `Training Duration`: **-0.0103** (virtually zero correlation).
  - `Training Cost` vs `Engagement Score`: **0.0214** (virtually zero correlation).
  - `Training Cost` vs `Satisfaction Score`: **-0.0029** (virtually zero correlation).
  - `Training Cost` vs `Work-Life Balance Score`: **0.0023** (virtually zero correlation).
* **Overall Pattern:** Every single correlation coefficient between the numerical variables falls between **-0.029 and +0.032**. This indicates that **none of the numerical features have any linear relationship with each other**. They behave as independent random distributions, which explains the low predictive capabilities of our machine learning models.

---

## 4. Explanation of the Models

### A. Linear Regression (OLS)
* **How it works:** Fits a linear equation ($y = \beta_0 + \beta_1 X_1 + ...$) by minimizing the sum of squared errors between actual and predicted scores.
* **Strengths:** Highly interpretable coefficients; extremely fast training.
* **Weaknesses:** Assumes linear relationships; cannot model complex patterns; outputs continuous float numbers that must be rounded for class ratings.

### B. Random Forest Regressor
* **How it works:** Fits an ensemble of 100 decision trees on bootstrapped training subsets and averages their predictions.
* **Strengths:** Captures complex non-linear combinations and interactions of features; robust to scale and outliers.
* **Weaknesses:** Poor interpretability ("black-box"); prone to overfitting on uncorrelated data; high memory footprint.

### C. Logistic Regression (Multi-class Classification)
* **How it works:** Calculates class probabilities for classes 1 to 5 using the multinomial (softmax) function.
* **Strengths:** Predicts probability distributions over ratings; very fast to train.
* **Weaknesses:** Assumes linear boundary lines; sensitive to unscaled inputs (converges slowly or fails without scaling); cannot model complex non-linear decision boundaries.

---

## 5. Performance Metrics Comparison

### Regression Models (Continuous Predictions)

| Metric | Linear Regression | Random Forest Regressor |
| :--- | :---: | :---: |
| **Mean Squared Error (MSE)** | 1.1041 | 1.1304 |
| **Root Mean Squared Error (RMSE)** | 1.0508 | 1.0632 |
| **R-squared ($R^2$) Score** | -0.0170 | -0.0412 |

* **Analysis:** The $R^2$ scores are negative/near-zero, showing that the features do not explain the target variance. 

### Classification Model (Discrete Predictions)

* **Model:** Logistic Regression
* **Classification Accuracy:** **53.33%** (320 correct predictions out of 600)

**Classification Report Summary:**
- Precision for class 3: **57%**
- Recall for class 3: **97%**
- Recall for class 1 and 5: **0%**
- **Analysis:** Since the features are uncorrelated with the target, the classification model minimizes its cross-entropy loss by predicting class `3` (the majority class representing 52% of the test set) for almost every employee.

---

## 6. Key Business Takeaways

1. **Weak Data Predictability:** Current captured HR variables do not predict employee ratings. Performance ratings are determined by other factors not captured here (e.g. manager relationship, personal motivation, project difficulty, specific KPIs, etc.).
2. **Reviewing Training & Satisfaction:** Training costs, duration, and satisfaction do not drive ratings. HR should re-evaluate how performance evaluations are structured and what metrics are used.
3. **Data Quality & Feature Engineering:** To build a predictive model that successfully forecasts performance or retention, HR must capture more relevant behavioral features, such as number of projects completed, average feedback scores, attendance rates, or peer reviews.

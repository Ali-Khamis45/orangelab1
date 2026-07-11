# HR Analytics: Machine Learning Methodology & Rationale Report

This report documents the scientific and statistical rationale behind the machine learning pipeline developed for predicting employee performance rating and attrition risk. It details the design decisions, feature engineering, validation strategies, and data cleaning workflows.

---

## 1. Target Variables Engineering

To extract maximum organizational value, we structured our targets around two primary business objectives:

### A. Performance Rating Prediction (`Current Employee Rating`)
- **Type:** Discrete Ordinal Score (1 to 5 scale).
- **Paradigm Rationale:** 
  - We model this variable using both **Regression** (predicting a continuous performance score spectrum) and **Classification** (predicting discrete reviewer buckets).
  - **Regression (continuous paradigm):** Treats ratings as numeric values, allowing linear and tree models to estimate intermediate performance grades (e.g., 3.25), which is useful for granular talent audits.
  - **Classification (categorical paradigm):** Treats ratings as distinct levels (1, 2, 3, 4, 5). This maps directly to standard HR review bands.
- **Limitation Disclosures:** Exploratory analysis indicates that the `Current Employee Rating` possesses very weak correlations with all employee features (coefficients ranging between -0.03 and +0.03). This represents a major dataset constraint, limiting performance prediction accuracy.

### B. Attrition Risk Prediction (`Is_Terminated`)
- **Type:** Binary Class Label (`1` for Terminated, `0` for Active).
- **Paradigm Rationale:**
  - This is modeled as a binary classification task to compute individual attrition probabilities. These probabilities are mapped to risk categories (Low, Moderate, High, Critical) in the dashboard, enabling HR managers to target retention efforts.

---

## 2. Feature Selection & Engineering

We structured our features across three dimensions to capture the employee lifecycle:

### A. Selected Features:
1. **Workplace Sentiment Indicators:** `Engagement Score`, `Satisfaction Score`, `Work-Life Balance Score`. These survey metrics reflect employee morale, which is statistically associated with job alignment and retention.
2. **Learning & Development (L&D) Metrics:** `Training Duration(Days)`, `Training Cost`, `Training Program Name`, `Training Type`. These capture corporate investments in skill development, which are expected to support higher performance ratings.
3. **Engineered Demographic & Lifecyle Metrics:**
   - `Age` (calculated using `Reference Date (2026-07-11) - DOB`): Age is a common demographic covariate associated with job stability.
   - `Tenure` (calculated using `ExitDate` or `Reference Date` minus `StartDate`): Tenure is a critical temporal factor associated with historical employee churn rates.
4. **Organizational Covariates:** `GenderCode`, `MaritalDesc`, `RaceDesc`, `DepartmentType`, `BusinessUnit`, `EmployeeType`, `PayZone`, `EmployeeClassificationType`. These control for systemic variations across departments and divisions.

### B. Excluded Features & Mathematical Rationale:
- **Identifiers:** `Employee ID`, `FirstName`, `LastName`, `ADEmail`. Excluded because they are unique high-cardinality keys. If kept, models (especially decision trees) would memorize individual records, causing overfitting and poor generalization.
- **Supervisor Name:** `Supervisor`. Excluded because it creates over 8,000 sparse dummy variables during encoding. This high-dimensionality sparse matrix slows down gradient solvers, induces severe overfitting, and leads to collinearity.
- **Null-Pruned Columns:** `Survey Date` (100% missing values) and `TerminationDescription` (leakage column since it is only populated after attrition has occurred).

---

## 3. Preprocessing, Encoding, & Standard Scaling

### Categorical One-Hot Encoding
- **Rationale:** Algorithms require numerical vectors to perform calculations. Categorical text (e.g., `Billing`, `Married`) is converted to binary indicator variables (`0` or `1`) using dummy encoding. We set `drop_first=True` to drop one category per column. This prevents multicollinearity (the dummy variable trap), which is critical for stabilizing coefficients in linear and logistic regression.

### Standard Scaling (`StandardScaler`)
- **Rationale:** Gradient and coordinate-descent solvers (like `lbfgs` or `saga` in Logistic Regression) navigate a cost surface to find optimal weights. If features have different scales (e.g., `Training Cost` is in thousands, while dummy variables are strictly `0` or `1`), the solver will take inefficient steps, oscillate, and may fail to converge.
- **Implementation:** Standard scaling scales numerical features to a mean of `0` and a standard deviation of `1`. This speeds up optimization and allows for direct comparison of feature coefficients.

---

## 4. Rigorous Validation Strategy

To prevent overfitting and ensure results are reliable, we replaced single train/test splits with a robust validation workflow:

- **Stratified K-Fold Cross-Validation (5 Folds):** Used for classification models. It partitions the dataset into 5 folds while preserving the ratio of positive and negative classes (attrition cases) in each fold. This ensures that validation scores are not biased by class imbalances.
- **K-Fold Cross-Validation (5 Folds):** Used for regression models. It evaluates the stability of RMSE and R² metrics across multiple data partitions.
- **Holdout Test Partition (20%):** A clean test set is kept completely isolated during all scaling and training steps. It is only used to compute final generalization metrics.

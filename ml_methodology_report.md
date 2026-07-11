# HR Analytics: Machine Learning Methodology & Rationale Report

This report explains the scientific and business methodology behind the machine learning pipeline designed for the HR dataset. It outlines the reasons for selecting the target, choosing specific features (and excluding others), applying standard scaling and encoding, and partitioning the dataset into train and test sets.

---

## 1. Why this Target Variable? (`Current Employee Rating`)

Our target variable is `Current Employee Rating` (Performance Rating score on a 1 to 5 scale).

### Rationale:
* **High Business Value:** Performance rating is the key metric that corporate HR departments and executive teams track to evaluate employee contributions, identify high potentials, and make promotional decisions. 
* **Analytic Flexibility:** Because the rating is a numeric score, it can be approached from two distinct machine learning paradigms:
  1. **Regression:** Treating the rating as a continuous number allows the models (Linear Regression, Random Forest) to predict intermediate values (e.g., 3.25) representing a continuous spectrum of performance quality.
  2. **Classification:** Treating the rating as a categorical class label (classes 1, 2, 3, 4, 5) allows the model (Logistic Regression) to predict the exact category probability, which is how HR departments evaluate employees on discrete scales.

---

## 2. Why these Features? (Feature Selection Rationale)

A total of 15 features were chosen because they provide a holistic view of the employee profile across three key dimensions:

### Selected Features:
1. **Employee Demographics & Profile:** `GenderCode`, `MaritalDesc`, `RaceDesc`. These features help audit and model whether personal background factors correlate with performance reviews (crucial for diversity, equity, and inclusion audits).
2. **Organizational Context:** `DepartmentType`, `BusinessUnit`, `EmployeeType`, `PayZone`, `EmployeeClassificationType`. These reflect the employee's work environment, pay level, and shift types, which affect workload and stress levels.
3. **Workplace Sentiment (Survey Scores):** `Engagement Score`, `Satisfaction Score`, `Work-Life Balance Score`. These survey metrics reflect employee morale and job alignment, which are theoretically directly tied to productivity.
4. **Learning & Development (Training Metrics):** `Training Program Name`, `Training Type`, `Training Duration(Days)`, `Training Cost`. These variables represent the company's investment in the employee's skill development, which is expected to boost performance.

### Excluded Features:
To prevent overfitting and ensure mathematical validity, the following features were excluded:
* **Unique Identifiers:** `Employee ID`, `FirstName`, `LastName`, `ADEmail`. These represent unique keys. If kept, models (especially Random Forests) will memorize individual IDs and overfit, failing to generalize to new employees.
* **Supervisor Name:** `Supervisor`. This was excluded to prevent bias toward specific supervisors and focus on systemic organizational factors.
* **Raw Timestamps:** `StartDate`, `ExitDate`, `DOB`, `Training Date`. Raw date strings cannot be parsed directly by machine learning algorithms. In a real-world setting, these would be engineered into numerical features (like `Tenure` or `Age`), but since they are not processed into numeric metrics in this notebook, they were excluded to avoid syntax errors.
* **Null-Pruned Columns:** `Survey Date` (dropped due to 100% missing values) and `TerminationDescription`.

---

## 3. How and Why did we Encode and Scale Features?

### Categorical One-Hot Encoding:
* **Why:** Machine learning algorithms perform mathematical dot products and vector calculations. Text categories (like `Billing` or `Married`) cannot be multiplied or added. 
* **How:** We used dummy encoding (`pd.get_dummies` with `drop_first=True`). This converts each categorical column into multiple binary (0 or 1) columns. Dropping the first category prevents multi-collinearity (the dummy variable trap), which is essential for stable coefficients in linear models.

### Standard Scaling (`StandardScaler`):
* **Why:** Gradient-descent and coordinate-descent solvers (like `lbfgs` used in Logistic Regression) search for the optimal model weights by navigating the error surface. If features have wildly different magnitudes—for example, `Training Cost` ranges in the hundreds/thousands while dummy variables are strictly 0 or 1—the optimization algorithm will take inefficient steps, oscillate, and fail to converge.
* **How:** Scaling transforms each feature so that it has a mean of 0 and standard deviation of 1. It normalizes the data distribution, guaranteeing rapid solver convergence and making model coefficients directly comparable in weight.

---

## 4. How and Why did we Split the Dataset?

### The 80% / 20% Split Ratio:
We split the 3,000-record dataset into:
- **80% Training Set (2,400 records):** This provides the models with a sufficiently large dataset to learn features, weights, and decision rules.
- **20% Testing Set (600 records):** This acts as an independent "holdout" set. The model is never shown this data during training. Evaluating on this unseen data is the only way to measure how well the models generalize to new hires.

### Fixed Random State (`random_state=42`):
* **Why:** Splitting data involves a random number generator. If we do not fix the seed, every time we run the notebook we will get a slightly different training set and test set. This makes model comparison impossible since the metrics (MSE, Accuracy) would fluctuate simply due to random partition chance.
* **The Choice of 42:** `42` is a standard, widely accepted seed value in the machine learning community (originally an homage to *The Hitchhiker's Guide to the Galaxy*). It guarantees that the data partition remains identical across different machines and executions.

---

## 5. Actionable Dashboard
The feature selection, scaling, and classification methodologies described above are deployed in the **Talent Attrition Advisor Web UI** (`static/index.html`). This allows HR users to dynamically test the impact of scaling, training, satisfaction, and tenure on real-time prediction probabilities.

# 📊 HR Employee Performance, Engagement & Talent Attrition Analytics

An end-to-end data cleaning, exploratory data analysis (EDA), and machine learning workflow analyzing 3,000+ employee records. This repository implements predictive models for employee performance and **talent attrition (churn)**, identifies key organizational drivers, and provides a web-based **Interactive Talent Attrition Advisor UI** for HR talent optimization.

<a href="https://colab.research.google.com/github/Ali-Khamis45/orangelab1/blob/main/taskLAB1orange.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
![Python Version](https://img.shields.style/badge/Python-3.12-blue.svg)
![License](https://img.shields.style/badge/License-MIT-green.svg)

---

## 🖥️ Interactive Web Dashboard (Advisor UI)
To make the machine learning models actionable for HR managers and business leaders, we built a premium web-based **Talent Attrition Advisor Dashboard**. This single-page application connects to a Flask backend serving the pre-trained, class-balanced machine learning model.

![Talent Attrition Advisor UI Dashboard](static/ui_screenshot.webp)

### Key Features:
* **Interactive Profile Form:** Input employee parameters (Age, Tenure, Satisfaction, Engagement, Work-Life Balance, Training Budgets, Departments, and Business Units).
* **Live Attrition Risk Gauge:** Displays the probability of churn (0% to 100%) in a smooth, color-coded gauge ring (Green for Low Risk, Coral for High Risk).
* **Relative Contribution Bars:** Breaks down how much different features (Tenure, Age, Training, Satisfaction) contribute to the overall churn risk.
* **Dynamic HR Recommendations:** Generates immediate, personalized intervention strategies based on the prediction results (e.g. scheduling early career syncs or job satisfaction audits).

---

## 📂 Repository Structure
* [taskLAB1orange.ipynb](file:///d:/orange/4/orangelab1/taskLAB1orange.ipynb): Complete Jupyter notebook containing full EDA, visualisations, and modeling.
* [server.py](file:///d:/orange/4/orangelab1/server.py): Flask backend server script serving predictions and UI assets.
* [static/index.html](file:///d:/orange/4/orangelab1/static/index.html): Premium, single-page web dashboard HTML, CSS, and JS.
* [ml_models/](file:///d:/orange/4/orangelab1/ml_models/): Serialized, pre-trained StandardScaler and balanced Logistic Regression attrition models.
* [ml_performance_report.md](file:///d:/orange/4/orangelab1/ml_performance_report.md): Comparative report on ratings regression and classification models.
* [ml_methodology_report.md](file:///d:/orange/4/orangelab1/ml_methodology_report.md): Detailed methodology documentation explaining feature/target selections, scaling, encoding, and splitting choices.
* [ml_attrition_report.md](file:///d:/orange/4/orangelab1/ml_attrition_report.md): Detailed classification report on predicting employee churn with ROC-AUC curves and feature importances.
* [cleaned_data_HR.csv](file:///d:/orange/4/orangelab1/cleaned_data_HR.csv): Cleaned intermediate dataset exported post-preprocessing.

---

## 🔄 End-to-End Analytics Workflow

### Phase 1: Data Audit & Cleaning
- Standardized custom string date formats (`StartDate`, `ExitDate`, `DOB`, `Training Date`) into standard Pandas `datetime64[ns]` objects.
- Pruned `Survey Date` due to >99% missing values and deduplicated records to eliminate statistical bias.
- Exported the results to `cleaned_data_HR.csv`.

### Phase 2: Exploratory Data Analysis (EDA) & Visuals
- Aggregated department performance scores (identifying **Billing** as the highest performing at `3.50` and **Sales** as the lowest at `2.72`).
- Designed publication-ready data plots:
  - **Department Headcount:** Donut and count charts.
  - **Evaluation Scores:** Styled rating histograms.
  - **Training Budgets:** Cost boxplots showing ranges and outliers.
  - **Retention Rates:** Employee status countplots grouped by Business Unit.

### Phase 3: Modeling Employee Ratings (Section 5)
- Encoded 10 categorical columns using one-hot dummy variables, yielding **38 input features**.
- Scaled features using `StandardScaler` to ensure solver convergence.
- Split data 80% / 20% (train/test) with a fixed random state (`42`).
- Trained and evaluated three models to predict `Current Employee Rating`:
  1. **Linear Regression (OLS)** ($R^2 \approx -0.0170$, $\text{RMSE} \approx 1.0508$)
  2. **Random Forest Regressor** ($R^2 \approx -0.0412$, $\text{RMSE} \approx 1.0632$)
  3. **Logistic Regression (Classification)** (Accuracy: **53.33%**)

### Phase 4: Correlation Matrix Heatmap (Section 6)
- Computed Pearson correlation coefficients for all numerical variables.
- Visualized correlations using a heatmap (`coolwarm` palette) and scatter plots (e.g. `Training Cost` vs. `Training Duration`).
- **Core Finding:** All numerical features have correlation values between **-0.029 and +0.032**. Since the variables are statistically independent, models predicting employee ratings from these features revert to predicting the baseline averages.

### Phase 5: Advanced Attrition Prediction (Section 7)
- **Advanced Feature Engineering:** Extracted `Age` and `Tenure` (employment duration in years) from raw timestamps.
- **Attrition Target:** Defined a binary target `Is_Terminated` (1 if terminated, 0 if active).
- **Class Balancing:** Addressed the 87% / 13% class imbalance using `class_weight='balanced'` in training.
- **Classification Modeling:** Trained balanced Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
- **Key Metrics:** The deployed Tuned Gradient Boosting model achieved a **recall of 75% for attrition** and an **ROC-AUC of 0.8078**, successfully flagging 75% of employees at risk of leaving!
- **Feature Importances & ROC Curves:** Plotted side-by-side ROC curve comparisons and a barplot showing the top 10 Gradient Boosting feature importances.

---

## 📈 Key Attrition Driver Insights (Random Forest)

The model identified the top structural drivers of employee churn:
1. **Tenure (20.08% Importance):** Employees are highly likely to leave at specific milestones in their lifecycle (e.g. after 1-2 years or when reaching retirement).
2. **Training Cost (10.55% Importance):** Suggests that employees who receive higher direct training investment are significantly better retained.
3. **Age (9.61% Importance):** Younger employees exhibit much higher baseline mobility than older, settled employees.
4. **Onboarding Duration (4.53% Importance):** Longer training duration correlates with higher retention.

---

## 🏢 Strategic HR Recommendations
1. **Design Milestone-Based Interventions:** Focus retention check-ins at critical tenure milestones (e.g. standard reviews at 1 year and 3 years) when employees are statistically most likely to leave.
2. **Tailor Mentorship for Younger Hires:** Since younger age is a primary driver of churn, establish clear career progression paths and peer mentorship for early-career hires.
3. **Implement Churn Dashboard:** Deploy the **Tuned Gradient Boosting Classifier model** as a retention warning tool. Since it successfully flags **75% of potential resignations**, HR can proactively offer salary reviews, work-life balance adjustments, or department changes before high-risk employees resign.

---

## ⚙️ Installation & Local Setup

To run the notebook and launch the interactive Web Advisor Dashboard locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ali-Khamis45/orangelab1.git
   cd orangelab1
   ```

2. **Install required libraries**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter flask
   ```

3. **Launch the interactive Advisor Web UI**:
   Run the Flask server:
   ```bash
   python server.py
   ```
   Open your browser and navigate to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

4. **Launch the notebook server (Optional)**:
   ```bash
   jupyter notebook
   ```
   Open `taskLAB1orange.ipynb` in your browser to inspect code cells.

# HR Employee Performance, Engagement & Training Analytics

An end-to-end data cleaning, exploratory data analysis (EDA), and advanced visualization workflow analyzing corporate HR records. This project identifies critical performance drivers, reviews training program costs, and provides data-driven recommendations for HR managers and business stakeholders.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Project Structure](#-project-structure)
3. [Installation & Setup](#-installation--setup)
4. [Workflow Execution Phases](#-workflow-execution-phases)
5. [Key Analytics & Insights](#-key-analytics--insights)
6. [Advanced Visualizations Summary](#-advanced-visualizations-summary)
7. [Business Recommendations](#-business-recommendations)

---

## 🔍 Overview
This project profiles employee records to extract actionable intelligence regarding corporate department performance, training efficiency, workforce status, and job satisfaction. The analysis is structured to ensure that data anomalies (missing values, improper types, duplicate entries) are systematically resolved before computing statistical insights.

Key highlights of the repository include:
* **Integrity First**: Standardized datetime objects, handled duplicates, and filtered out high-null columns (`Survey Date`).
* **Granular Aggregations**: Explored metrics across genders, departments, and business units.
* **Production-Grade Visuals**: Created publication-ready data plots including custom donut charts, box plots, and styled histograms.
* **Actionable Rationale**: In-depth text annotations explaining the "why" and "how" behind every line of code.

---

## 📂 Project Structure
```bash
├── taskLAB1orange.ipynb       # Documented Jupyter Notebook containing full EDA & code cells
├── cleaned_data_HR.csv        # Cleaned dataset exported post-preprocessing
├── documentation.txt          # In-depth analysis report of metrics and cleaning methodologies
└── implementation_details.txt # Technical documentation of execution flow and dependencies
```

---

## ⚙️ Installation & Setup

To execute the notebook and view the interactive visualizations locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ali-Khamis45/orangelab1.git
   cd orangelab1
   ```

2. **Install required libraries**:
   Ensure you have Python 3 installed. Install dependencies using `pip`:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. **Launch the notebook server**:
   ```bash
   jupyter notebook
   ```
   Open `taskLAB1orange.ipynb` in your browser.

---

## 🔄 Workflow Execution Phases

### Phase 1: Data Audit & Cleansing
* Loaded the raw `HR_Dataset.csv` dataset.
* Checked dataset dimensions (columns/rows), surveyed data types, and compiled missing values.
* Pruned the `Survey Date` column (missing >99% of values).
* Converted string columns (`StartDate`, `ExitDate`, `DOB`, `Training Date`) into standard Pandas `datetime64[ns]` objects.
* Audited and dropped all duplicate records to avoid statistical bias.
* Exported the resulting dataframe to `cleaned_data_HR.csv`.

### Phase 2: Descriptive & Grouped Aggregations
* Evaluated corporate baseline stats (average employee rating: `2.97`, average engagement: `2.94`).
* Calculated mean performance ratings across departments and satisfaction across business units.
* Analyzed average budget spending on different corporate training programs.
* Checked for gender differences in work-life balance scores.

### Phase 3: Visual Analytics
* **Donut Chart**: Shows the distribution of employees across departments.
* **Countplots**: Displays department sizes and status (Active/Terminated) by Business Unit.
* **Histograms**: Plots the shape of employee performance ratings.
* **Boxplots**: Compares training program costs, highlighting outliers and budget ranges.

---

## 💡 Key Analytics & Insights
* **Top Department by Performance Rating**: The **Billing** department holds the highest average performance rating (`3.50`), whereas the **Sales** department holds the lowest (`2.72`).
* **Morale vs. Satisfaction**: There is a weak positive Pearson correlation (`0.0167`) between employee `Engagement Score` and `Satisfaction Score`, showing that engagement and satisfaction do not necessarily move in lockstep.
* **Training and Performance**: There is a small negative correlation (`-0.0219`) between the training duration (in days) and the employee's final performance rating, suggesting that longer training courses do not automatically lead to better job performance ratings.

---

## 📈 Advanced Visualizations Summary
The project includes several key charts built using Seaborn and Matplotlib:
1. **Department Type Distribution**: A clean donut chart showing the percentage of employees in each department.
2. **Current Employee Rating Distribution**: A polished histogram showing performance evaluation scores.
3. **Training Cost Distribution by Program**: A horizontal boxplot showing cost distributions, median spends, and outliers across programs.
4. **Employee Status by Business Unit**: A grouped horizontal countplot displaying active versus terminated employees in each business division.

---

## 🏢 Business Recommendations
1. **Investigate Training Curriculum**: Since longer training durations do not correlate with higher performance ratings, review course content to ensure it aligns with day-to-day employee responsibilities.
2. **Review Sales Department Support**: Since the Sales department has the lowest average employee performance rating, HR should review sales training, targets, and supervisor support.
3. **Analyze Business Unit Retention**: Use the Business Unit countplot to identify divisions with high termination rates and target them for retention strategies.

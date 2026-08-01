# Task 1: Credit Scoring Model

## Objective
Predict an individual's creditworthiness (likelihood of defaulting on debt within 2 years) using historical financial data, framed as a binary classification problem.

## Dataset
**Give Me Some Credit** (Kaggle) — [link](https://www.kaggle.com/c/GiveMeSomeCredit/data)

- 150,000 records, 10 features + 1 target (`SeriousDlqin2yrs`)
- Target distribution: 93.3% non-default, 6.7% default (imbalanced)
- Features include: revolving credit utilization, age, monthly income, debt ratio, number of open credit lines, and counts of past-due payments (30-59, 60-89, and 90+ days)

Place `cs-training.csv` in `data/` before running the notebook (not included in this repo due to size/licensing — download from the link above).

## Approach

### 1. Data Cleaning
- Fixed placeholder error codes (98) in the three "days past due" columns, capped at the true max value
- Corrected `age = 0` entries (data error) using the median
- Capped `RevolvingUtilizationOfUnsecuredLines` at 2 (values above this are implausible for a ratio)
- Imputed missing `MonthlyIncome` (~20% missing) and `NumberOfDependents` (~2.6% missing) using median/mode

### 2. Feature Engineering
- `TotalPastDue`: sum of all late-payment counts across the three due-date columns
- `IncomePerDependent`: monthly income adjusted for number of dependents
- `CreditLinesPerAge`: number of open credit lines relative to age
- `HasRealEstateLoan`: binary flag for real estate loan ownership

### 3. Modeling
Trained and compared three classifiers on an 80/20 stratified train/test split:
- **Logistic Regression** (baseline, features scaled with `StandardScaler`)
- **Decision Tree** (`max_depth=10`)
- **Random Forest** (`n_estimators=200`, `max_depth=10`)

All models used `class_weight='balanced'` to address the 93/7 class imbalance.

### 4. Evaluation
Since the dataset is heavily imbalanced, accuracy alone is misleading (a model predicting "no default" for everyone would score ~93%). Models were evaluated primarily on **Recall** (catching actual defaulters), **Precision**, **F1-score**, and **ROC-AUC**.

## Results

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Accuracy | 0.80 | 0.79 | **0.82** |
| Recall (default class) | 0.75 | 0.75 | 0.75 |
| Precision (default class) | 0.21 | 0.20 | **0.24** |
| F1-score (default class) | 0.33 | 0.32 | **0.36** |
| ROC-AUC | 0.859 | 0.818 | **0.867** |

**Random Forest performed best across every metric** and was selected as the final model, saved to `models/random_forest_credit_model.pkl`.

## Key Findings
- `RevolvingUtilizationOfUnsecuredLines` and `TotalPastDue` showed the clearest separation between defaulters and non-defaulters, and were the strongest predictors.
- Increasing Random Forest `max_depth` from 10 to 100 raised accuracy (0.82 → 0.92) but caused **overfitting**: recall on the default class dropped sharply (0.75 → 0.37) and ROC-AUC decreased (0.867 → 0.847). This confirmed that accuracy is not a reliable metric on imbalanced data, and `max_depth=10` was kept as the better-generalizing choice.
- A single Decision Tree underperformed the ensemble Random Forest, as expected — individual trees are more prone to variance/overfitting than an averaged ensemble.

## Project Structure
```
Task1/
├── data/           # raw dataset (not committed — download separately)
├── notebooks/
│   └── eda.ipynb   # full EDA, cleaning, feature engineering, modeling, evaluation
├── models/
│   └── random_forest_credit_model.pkl
└── README.md
```

## How to Run
```bash
# from repo root
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```
Place `cs-training.csv` in `Task1/data/`, then open `Task1/notebooks/eda.ipynb` in VS Code (Jupyter extension) and run all cells.

## Tools
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

## Author
CodeAlpha Machine Learning Internship — Task 1
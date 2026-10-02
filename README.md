# Credit Wise Loan System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-blue?logo=python">
  <img src="https://img.shields.io/badge/Machine-Learning-orange">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
</p>

<p align="center">
  A Machine Learning project for loan approval prediction
  using applicant and financial information.
</p>

A machine-learning classification project that explores applicant and financial information to predict the `Loan_Approved` target.

## Project overview

This project uses a Jupyter Notebook to perform data cleaning, exploratory data analysis (EDA), categorical encoding, feature scaling, model training, and evaluation for a loan approval classification task.

## Dataset

- File: `loan_approval_data.csv`
- Rows: 1,000
- Columns: 20
- Target column: `Loan_Approved`

The dataset contains applicant and loan-related fields such as applicant income, co-applicant income, employment status, age, credit score, existing loans, debt-to-income ratio, savings, collateral value, loan amount, loan term, loan purpose, property area, education level, gender, and employer category.

## Workflow in the notebook

1. Load the dataset with Pandas.
2. Inspect the data and handle missing values:
   - Numerical columns: mean imputation.
   - Categorical columns: most-frequent-value imputation.
3. Explore the data using Matplotlib and Seaborn visualizations.
4. Drop the `Applicant_ID` field.
5. Encode categorical features using Label Encoding and One-Hot Encoding.
6. Review feature correlations.
7. Split the data into training and test sets (80:20, `random_state=42`).
8. Standardize features using `StandardScaler`.
9. Train and evaluate:
   - Logistic Regression
   - K-Nearest Neighbors (kNN, k=5)
   - Gaussian Naive Bayes
10. Add squared features for `DTI_Ratio` and `Credit_Score`, then repeat model training and evaluation.

## Technologies and libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Repository structure

```text
Credit-Wise-Loan-System/
├── CreditWise_Loan_System(2).ipynb
├── loan_approval_data.csv
├── README.md
└── requirements.txt
```

## Run locally

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `CreditWise_Loan_System.ipynb` and run the cells from top to bottom.

Keep the notebook and CSV file in the same folder so that `pd.read_csv("loan_approval_data.csv")` can find the dataset.

 ## Model Evaluation

Three machine learning models were evaluated before and after feature engineering using Accuracy, Precision, Recall, and F1-score.

### Performance After Feature Engineering

| Model                     | Accuracy | Precision | Recall | F1-Score |
| ------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression       |    87.5% |    79.03% | 80.33% |   79.67% |
| K-Nearest Neighbors (KNN) |    75.5% |    62.00% | 50.82% |   55.86% |
| Gaussian Naive Bayes      |    86.5% |    78.33% | 77.05% |   77.69% |

Based on the reported test results, Logistic Regression achieved the highest accuracy, recall, and F1-score after feature engineering, and was selected as the final model.

These results are specific to this dataset and test split. Further validation is required before using the model for real-world decisions.

## Important note

This is an educational machine-learning project. Its predictions should not be used as the sole basis for real loan approval or credit decisions. Real lending decisions require validated data, fairness checks, explainability, regulatory compliance, and human review.

## Author

**Puneet Narayan**


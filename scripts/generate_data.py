"""
generate_data.py - Smart Lender Dataset Generator
Generates a realistic synthetic loan prediction dataset modeled after
the standard Kaggle Loan Prediction Dataset structure.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 614  # mimic Kaggle dataset size

def generate_dataset(n=N):
    data = {}

    # Gender
    data['Gender'] = np.random.choice(['Male', 'Female'], size=n, p=[0.80, 0.20])

    # Married
    data['Married'] = np.random.choice(['Yes', 'No'], size=n, p=[0.65, 0.35])

    # Dependents
    data['Dependents'] = np.random.choice(['0', '1', '2', '3+'], size=n, p=[0.57, 0.17, 0.16, 0.10])

    # Education
    data['Education'] = np.random.choice(['Graduate', 'Not Graduate'], size=n, p=[0.78, 0.22])

    # Self_Employed
    data['Self_Employed'] = np.random.choice(['Yes', 'No'], size=n, p=[0.14, 0.86])

    # ApplicantIncome (log-normal)
    data['ApplicantIncome'] = np.random.lognormal(mean=8.5, sigma=0.6, size=n).astype(int)
    data['ApplicantIncome'] = np.clip(data['ApplicantIncome'], 1000, 81000)

    # CoapplicantIncome
    data['CoapplicantIncome'] = np.where(
        np.random.random(n) < 0.45,
        np.random.lognormal(mean=7.5, sigma=0.7, size=n).astype(int),
        0
    )

    # LoanAmount (in thousands)
    data['LoanAmount'] = np.random.lognormal(mean=4.9, sigma=0.5, size=n).astype(int)
    data['LoanAmount'] = np.clip(data['LoanAmount'], 9, 700)

    # Loan_Amount_Term (months)
    data['Loan_Amount_Term'] = np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                                 size=n, p=[0.01, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.05, 0.68, 0.08])

    # Credit_History (1 = good, 0 = bad)
    data['Credit_History'] = np.random.choice([1.0, 0.0], size=n, p=[0.84, 0.16])

    # Property_Area
    data['Property_Area'] = np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n, p=[0.38, 0.35, 0.27])

    df = pd.DataFrame(data)

    # Generate Loan_Status based on realistic credit rules
    approved_prob = np.zeros(n)

    # Credit history is the strongest predictor
    approved_prob += np.where(df['Credit_History'] == 1.0, 0.55, -0.30)

    # Income ratio
    total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
    emi_estimate = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
    income_ratio = emi_estimate / (total_income + 1)
    approved_prob += np.where(income_ratio < 0.3, 0.15, np.where(income_ratio < 0.5, 0.05, -0.15))

    # Education
    approved_prob += np.where(df['Education'] == 'Graduate', 0.10, -0.05)

    # Self Employed (slight penalty)
    approved_prob += np.where(df['Self_Employed'] == 'Yes', -0.05, 0.05)

    # Married (slight positive)
    approved_prob += np.where(df['Married'] == 'Yes', 0.05, 0.0)

    # Property Area
    approved_prob += np.where(df['Property_Area'] == 'Semiurban', 0.08,
                              np.where(df['Property_Area'] == 'Urban', 0.05, 0.0))

    # Gender
    approved_prob += np.where(df['Gender'] == 'Male', 0.02, 0.0)

    # Clip to [0.05, 0.95] and sample
    approved_prob = np.clip(approved_prob + 0.55, 0.05, 0.95)
    df['Loan_Status'] = np.where(np.random.random(n) < approved_prob, 'Y', 'N')

    # Introduce ~5% missing values to simulate real data
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        missing_mask = np.random.random(n) < 0.04
        df.loc[missing_mask, col] = np.nan

    # Add Loan_ID
    df.insert(0, 'Loan_ID', [f'LP{str(i).zfill(6)}' for i in range(1, n+1)])

    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_dataset()
    df.to_csv('data/loan_data.csv', index=False)
    print(f"Dataset generated: {len(df)} rows")
    print(f"Loan Status distribution:\n{df['Loan_Status'].value_counts()}")
    print(f"Saved to data/loan_data.csv")

import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000  # number of records

# Generate base features
income = np.random.normal(75000, 20000, n).clip(20000, 200000)
debt = np.random.normal(20000, 10000, n).clip(0, 100000)
credit_score = np.random.normal(680, 50, n).clip(300, 850)
loan_amount = np.random.normal(25000, 15000, n).clip(5000, 100000)

loan_purpose = np.random.choice(
    ['Business', 'Personal', 'Real Estate', 'Education'], n
)

# Create DataFrame
df = pd.DataFrame({
    'Income': income,
    'Debt': debt,
    'CreditScore': credit_score,
    'LoanAmount': loan_amount,
    'LoanPurpose': loan_purpose
})

# Feature Engineering
df['DTI'] = df['Debt'] / df['Income']

# Risk Score (0–1 scale)
df['RiskScore'] = (
    0.5 * df['DTI'] +
    0.3 * (1 - df['CreditScore'] / 850) +
    0.2 * (df['LoanAmount'] / 100000)
)

# Default Probability (sigmoid function)
df['DefaultProbability'] = 1 / (1 + np.exp(-5 * (df['RiskScore'] - 0.5)))

# Risk Category
df['RiskCategory'] = pd.cut(
    df['RiskScore'],
    bins=[0, 0.3, 0.6, 1],
    labels=['Low', 'Medium', 'High']
)

# Save dataset
df.to_csv('loan_data.csv', index=False)

print(df.head())

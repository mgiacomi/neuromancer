import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()

# Combine features and target into a single pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

# Save to a CSV file
df.to_csv('breast_cancer_wisconsin.csv', index=False)
print("CSV successfully created as 'breast_cancer_wisconsin.csv'!")
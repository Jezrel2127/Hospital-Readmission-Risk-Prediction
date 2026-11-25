import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data.csv"


print("HOSPITAL READMISSION RISK PREDICTION")
print("=" * 50)

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("\n1. DATASET INFORMATION")
print("-" * 50)
print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])
print("\n2. FIRST FIVE RECORDS")
print("-" * 50)
print(df.head())
print("\n3. DATASET COLUMNS")
print("-" * 50)
print(df.columns.tolist())
print("\n4. MISSING VALUES")
print("-" * 50)
print(df.isnull().sum().sort_values(ascending=False).head(15))
print("\n5. DUPLICATE RECORDS")
print("-" * 50)
print("Duplicates:", df.duplicated().sum())
print("\n6. READMISSION DISTRIBUTION")
print("-" * 50)
print(df["readmitted"].value_counts())
print("\nDATASET EXPLORATION COMPLETED!")

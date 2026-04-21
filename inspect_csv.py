import pandas as pd

df = pd.read_csv("data/NHANES/nhanes_diabetes.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

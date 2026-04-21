import pandas as pd
import numpy as np

path = "data/NHANES/"

demo = pd.read_sas(path + "DEMO_J.xpt", format="xport")
bmx  = pd.read_sas(path + "BMX_J.xpt", format="xport")   # BMI
bp   = pd.read_sas(path + "BPX_J.xpt", format="xport")    # Blood pressure
diq  = pd.read_sas(path + "DIQ_J.xpt", format="xport")    # Diabetes questionnaire

print("Loaded files successfully.")

df = demo.merge(bmx, on="SEQN", how="inner") \
         .merge(bp, on="SEQN", how="inner") \
         .merge(diq, on="SEQN", how="inner")

print("Merged shape:", df.shape)

cols = [
    "SEQN",
    "RIDAGEYR",   # age
    "RIAGENDR",   # sex
    "RIDRETH3",   # ethnicity
    "BMXBMI",     # BMI
    "BPXSY1",     # systolic BP
    "BPXDI1",     # diastolic BP
    "DIQ010"      # diabetes yes/no
]

df = df[cols]

df = df.replace([7, 9, 77, 99, 7777, 9999, "."], np.nan)

df = df[df["DIQ010"].isin([1, 2])]  # keep only yes/no answers
df["Diabetes_binary"] = df["DIQ010"].map({1: 1, 2: 0})

df = df.dropna(subset=["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "BPXSY1", "BPXDI1"])

df = df[(df["BMXBMI"] >= 10) & (df["BMXBMI"] <= 80)]
df = df[(df["BPXSY1"] >= 80) & (df["BPXSY1"] <= 250)]
df = df[(df["BPXDI1"] >= 40) & (df["BPXDI1"] <= 150)]
df = df[(df["RIDAGEYR"] >= 18) & (df["RIDAGEYR"] <= 80)]

print("After cleaning:", df.shape)

df.to_csv("nhanes_cleaned_clinical.csv", index=False)
print("Saved cleaned dataset as nhanes_cleaned_clinical.csv")

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import joblib

# Load model + scaler
model = joblib.load("clinical_model.pkl")
scaler = joblib.load("clinical_scaler.pkl")

# Load dataset
df = pd.read_csv("nhanes_cleaned_clinical.csv")

# Prepare features
X = df[["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "BPXSY1", "BPXDI1"]]
y = df["Diabetes_binary"]

# Encode sex
X["RIAGENDR"] = X["RIAGENDR"].map({1: 0, 2: 1})

# One-hot encode ethnicity
X = pd.get_dummies(X, columns=["RIDRETH3"], drop_first=True)

# Scale
X_scaled = scaler.transform(X)

# Predict probabilities
df["pred_proba"] = model.predict_proba(X_scaled)[:, 1]

# Function to compute metrics at any threshold
def evaluate_threshold(threshold):
    df["pred"] = (df["pred_proba"] >= threshold).astype(int)

    results = {}

    # Overall metrics
    results["overall_accuracy"] = accuracy_score(y, df["pred"])
    results["overall_recall"] = recall_score(y, df["pred"])
    results["overall_precision"] = precision_score(y, df["pred"])

    # Metrics by ethnicity
    ethnic_groups = df["RIDRETH3"].unique()

    group_metrics = {}

    for group in ethnic_groups:
        subset = df[df["RIDRETH3"] == group]

        if len(subset) == 0:
            continue

        group_metrics[int(group)] = {
            "accuracy": accuracy_score(subset["Diabetes_binary"], subset["pred"]),
            "recall": recall_score(subset["Diabetes_binary"], subset["pred"]),
            "precision": precision_score(subset["Diabetes_binary"], subset["pred"]),
            "false_negative_rate": 1 - recall_score(subset["Diabetes_binary"], subset["pred"])
        }

    results["groups"] = group_metrics

    # Fairness metrics
    recalls = [m["recall"] for m in group_metrics.values()]
    fnrs = [m["false_negative_rate"] for m in group_metrics.values()]

    results["fairness_gap_recall"] = max(recalls) - min(recalls)
    results["fairness_gap_fnr"] = max(fnrs) - min(fnrs)

    return results

# Test at default threshold
metrics = evaluate_threshold(0.5)
print(metrics)

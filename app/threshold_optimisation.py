import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

model = joblib.load(os.path.join(MODEL_DIR, "clinical_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "clinical_scaler.pkl"))
feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "clinical_feature_columns.pkl")
)

df = pd.read_csv(os.path.join(DATA_DIR, "nhanes_cleaned_clinical.csv"))

df = df.rename(columns={
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "BPXSY1": "SystolicBP",
    "BPXDI1": "DiastolicBP",
    "RIAGENDR": "Sex",
    "RIDRETH3": "Ethnicity",
    "Diabetes_binary": "Diabetes",
})

# Match the training pipeline: Sex mapped to 0/1, Ethnicity one-hot encoded
df["Sex"] = df["Sex"].map({1: 0, 2: 1})

TARGET_COL = "Diabetes"
GROUP_COL = "Ethnicity"  # kept untouched for fairness grouping

# Preserve the raw ethnicity code for fairness grouping before one-hot encoding
groups = df[GROUP_COL].copy()
y = df[TARGET_COL].astype(int).values

# Build features the same way the training script did
NUMERIC_COLS = ["Age", "BMI", "SystolicBP", "DiastolicBP"]
X = df[NUMERIC_COLS + ["Sex", "Ethnicity"]].copy()
X = pd.get_dummies(X, columns=["Ethnicity"], drop_first=True)

# Align columns to the saved feature order, filling any missing columns with 0
X = X.reindex(columns=feature_columns, fill_value=0)

# Only the numeric columns were scaled during training — do the same here
X_arr = X.to_numpy(dtype=float)
numeric_idx = [feature_columns.index(c) for c in NUMERIC_COLS]
X_arr[:, numeric_idx] = scaler.transform(X_arr[:, numeric_idx])


def evaluate_at_threshold(threshold: float) -> dict:
    probs = model.predict_proba(X_arr)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    overall_accuracy = accuracy_score(y, y_pred)
    overall_recall = recall_score(y, y_pred)
    group_metrics = {}
    recalls = []

    for g in sorted(groups.unique()):
        mask = (groups == g)
        y_true_g = y[mask]
        y_pred_g = y_pred[mask]

        if y_true_g.sum() == 0:
            group_recall = np.nan
        else:
            group_recall = recall_score(y_true_g, y_pred_g)

        group_metrics[int(g)] = {
            "recall": float(group_recall) if not np.isnan(group_recall) else None,
            "support": int(len(y_true_g)),
        }

        if not np.isnan(group_recall):
            recalls.append(group_recall)

    if len(recalls) > 0:
        fairness_gap_recall = float(np.max(recalls) - np.min(recalls))
    else:
        fairness_gap_recall = 0.0

    return {
        "threshold": float(threshold),
        "overall_accuracy": float(overall_accuracy),
        "overall_recall": float(overall_recall),
        "fairness_gap_recall": fairness_gap_recall,
        "groups": group_metrics,
    }


def sweep_thresholds(thresholds):
    """Evaluate a list of thresholds and return metrics for each."""
    return [evaluate_at_threshold(t) for t in thresholds]


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for t in test_thresholds:
        m = evaluate_at_threshold(t)
        print(
            f"t={t:.2f} | acc={m['overall_accuracy']:.3f} "
            f"| recall={m['overall_recall']:.3f} "
            f"| gap={m['fairness_gap_recall']:.3f}"
        )

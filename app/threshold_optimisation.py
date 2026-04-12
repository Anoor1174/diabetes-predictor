import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")  # nhanes_cleaned_clinical.csv is in /data

model = joblib.load(os.path.join(MODEL_DIR, "clinical_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "clinical_scaler.pkl"))

df = pd.read_csv(os.path.join(DATA_DIR, "nhanes_cleaned_clinical.csv"))


df = df.rename(columns={
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "BPXSY1": "SystolicBP",
    "BPXDI1": "DiastolicBP",
    "RIAGENDR": "Sex",
    "RIDRETH3": "Ethnicity",
    "Diabetes_binary": "Diabetes"
})


FEATURES = ["SystolicBP", "DiastolicBP", "BMI", "Age", "Sex", "Ethnicity"]
TARGET_COL = "Diabetes"  
GROUP_COL = "Ethnicity"  # fairness group

# Ensure required columns exist
missing = [c for c in FEATURES + [TARGET_COL, GROUP_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in dataframe: {missing}")

X = df[FEATURES].copy()
y = df[TARGET_COL].astype(int).values
groups = df[GROUP_COL].copy()

X_scaled = scaler.transform(X)


def evaluate_at_threshold(threshold: float) -> dict:
    """
    Evaluate model performance and fairness at a given decision threshold.
    Returns a dict with:
      - overall_accuracy
      - overall_recall
      - fairness_gap_recall
      - groups: {group_name: {recall, support}}
    """
    # Predicted probabilities for positive class
    probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    overall_accuracy = accuracy_score(y, y_pred)
    overall_recall = recall_score(y, y_pred)

    # Group-wise recall
    group_metrics = {}
    recalls = []

    for g in sorted(groups.unique()):
        mask = (groups == g)
        y_true_g = y[mask]
        y_pred_g = y_pred[mask]

        if y_true_g.sum() == 0:
            # No positives in this group → recall undefined; skip from fairness gap
            group_recall = np.nan
        else:
            group_recall = recall_score(y_true_g, y_pred_g)

        group_metrics[g] = {
            "recall": float(group_recall) if not np.isnan(group_recall) else None,
            "support": int(len(y_true_g))
        }

        if not np.isnan(group_recall):
            recalls.append(group_recall)

    # Fairness gap = max recall - min recall across groups with defined recall
    if len(recalls) > 0:
        fairness_gap_recall = float(np.max(recalls) - np.min(recalls))
    else:
        fairness_gap_recall = 0.0

    return {
        "threshold": float(threshold),
        "overall_accuracy": float(overall_accuracy),
        "overall_recall": float(overall_recall),
        "fairness_gap_recall": fairness_gap_recall,
        "groups": group_metrics
    }

def sweep_thresholds(thresholds: list[float]) -> list[dict]:
    """
    Evaluate a list of thresholds and return metrics for each.
    """
    results = []
    for t in thresholds:
        metrics = evaluate_at_threshold(t)
        results.append(metrics)
    return results


if __name__ == "__main__":
    # Quick manual test
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for t in test_thresholds:
        m = evaluate_at_threshold(t)
        print(
            f"t={t:.2f} | acc={m['overall_accuracy']:.3f} "
            f"| recall={m['overall_recall']:.3f} "
            f"| gap={m['fairness_gap_recall']:.3f}"
        )


import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Possible locations of the cleaned dataset
CANDIDATE_DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "nhanes_cleaned_clinical.csv"),
    os.path.join(BASE_DIR, "nhanes_cleaned_clinical.csv"),
]
# Output path for results CSV
OUTPUT_PATH = os.path.join(BASE_DIR, "model_comparison_results.csv")

# Screening threshold prioritising recall
THRESHOLD = 0.15
# Fixed seed for reproducibility
RANDOM_STATE = 42
# Number of folds for cross-validation
N_SPLITS = 5

# Numeric columns that require scaling
NUMERIC_COLS = ["Age", "BMI", "SystolicBP", "DiastolicBP"]
# Column holding the target labels
TARGET_COL = "Diabetes"


def resolve_data_path():
    # Return the first path that exists
    for path in CANDIDATE_DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cleaned dataset not found in {CANDIDATE_DATA_PATHS}. "
        "Run data_preprocessing.py first."
    )


def load_data():
    # Load and rename columns to friendly names
    df = pd.read_csv(resolve_data_path()).rename(columns={
        "RIDAGEYR": "Age",
        "BMXBMI": "BMI",
        "BPXSY1": "SystolicBP",
        "BPXDI1": "DiastolicBP",
        "RIAGENDR": "Sex",
        "RIDRETH3": "Ethnicity",
        "Diabetes_binary": "Diabetes",
    })

    # Recode sex to 0/1
    df["Sex"] = df["Sex"].map({1: 0, 2: 1})

    # Select features used by the model
    X = df[NUMERIC_COLS + ["Sex", "Ethnicity"]].copy()
    # One-hot encode ethnicity
    X = pd.get_dummies(X, columns=["Ethnicity"], drop_first=True)
    # Convert target to integer labels
    y = df[TARGET_COL].astype(int)
    return X, y


def score_at_threshold(y_true, probs, threshold):
    # Apply threshold to probabilities
    preds = (probs >= threshold).astype(int)
    # Return standard classification metrics
    return {
        "auc": roc_auc_score(y_true, probs),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
    }


def evaluate_cv(name, estimator_factory, X, y):
    # Stratified K-fold preserves class balance
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    # Index positions of numeric columns
    numeric_idx = [X.columns.get_loc(c) for c in NUMERIC_COLS]
    # Work with numpy arrays for speed
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy()

    # Collect fold-level scores
    fold_scores = {"auc": [], "accuracy": [], "precision": [], "recall": []}

    # Iterate over each fold
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        # Split into train and test
        X_train = X_arr[train_idx].copy()
        X_test = X_arr[test_idx].copy()
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_train[:, numeric_idx] = scaler.fit_transform(X_train[:, numeric_idx])
        X_test[:, numeric_idx] = scaler.transform(X_test[:, numeric_idx])

        # Apply SMOTE to training data only
        X_train_res, y_train_res = SMOTE(
            random_state=RANDOM_STATE
        ).fit_resample(X_train, y_train)

        # Fit the candidate model
        model = estimator_factory()
        model.fit(X_train_res, y_train_res)
        # Predicted probabilities on the test fold
        probs = model.predict_proba(X_test)[:, 1]

        # Record metrics for this fold
        for metric, value in score_at_threshold(y_test, probs, THRESHOLD).items():
            fold_scores[metric].append(value)

    # Return mean and std across folds
    summary = {"model": name}
    for metric, values in fold_scores.items():
        summary[f"{metric}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{metric}_std"] = round(float(np.std(values)), 4)
    return summary


def main():
    # Load the features and target
    X, y = load_data()
    # Candidate models to compare
    model_factories = [
        ("LogisticRegression", lambda: LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        )),
        ("RandomForest", lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ("XGBoost", lambda: XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ]

    # Collect results for each model
    records = []
    for name, factory in model_factories:
        print(f"Running {name} ({N_SPLITS}-fold CV)...")
        record = evaluate_cv(name, factory, X, y)
        records.append(record)
        print(
            f"  recall@{THRESHOLD}: "
            f"{record['recall_mean']:.3f} ± {record['recall_std']:.3f}   "
            f"precision@{THRESHOLD}: "
            f"{record['precision_mean']:.3f} ± {record['precision_std']:.3f}   "
            f"AUC: {record['auc_mean']:.3f}"
        )

    # Save results to CSV for the report
    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
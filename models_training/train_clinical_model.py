import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Possible locations of the cleaned dataset
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "nhanes_cleaned_clinical.csv"),
    os.path.join(BASE_DIR, "nhanes_cleaned_clinical.csv"),
]
# Output paths for saved artefacts
MODEL_PATH = os.path.join(BASE_DIR, "clinical_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "clinical_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "clinical_feature_columns.pkl")
TEST_SET_PATH = os.path.join(BASE_DIR, "clinical_test_set.csv")

# Screening threshold prioritising recall
THRESHOLD = 0.15
# Fixed seed for reproducibility
RANDOM_STATE = 42
# Number of folds for cross-validation
N_SPLITS = 5
# Numeric columns that require scaling
NUMERIC_COLS = ["Age", "BMI", "SystolicBP", "DiastolicBP"]


def resolve_data_path():
    # Return the first path that exists
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Run data_preprocessing.py first to produce "
        "nhanes_cleaned_clinical.csv."
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
    # One-hot encode ethnicity — NHANES codes are nominal, not ordinal.
    X = pd.get_dummies(X, columns=["Ethnicity"], drop_first=True)
    # Convert target to integer labels
    y = df["Diabetes"].astype(int)
    return X, y


def cv_recall_at_threshold(estimator_factory, X, y, threshold=THRESHOLD):
    """5-fold stratified CV recall at a given threshold.

    Scaling and SMOTE are fit inside each fold.
    """
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
    recalls, precisions, aucs = [], [], []
    # Iterate over each fold
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        # Split into train and test
        X_train, X_test = X_arr[train_idx].copy(), X_arr[test_idx].copy()
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_train[:, numeric_idx] = scaler.fit_transform(X_train[:, numeric_idx])
        X_test[:, numeric_idx] = scaler.transform(X_test[:, numeric_idx])

        # Apply SMOTE to training data only
        X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
            X_train, y_train
        )
        # Fit the candidate model
        model = estimator_factory()
        model.fit(X_res, y_res)
        # Predicted probabilities on the test fold
        probs = model.predict_proba(X_test)[:, 1]
        # Apply screening threshold
        preds = (probs >= threshold).astype(int)

        # Record metrics for this fold
        recalls.append(recall_score(y_test, preds))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        aucs.append(roc_auc_score(y_test, probs))

    # Return mean and std across folds
    return {
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "auc_mean": float(np.mean(aucs)),
    }


def fit_final_model(estimator_factory, X, y):
    """Fit the selected model on a fixed train split, save held-out test."""
    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Fit scaler on training set
    scaler = StandardScaler()
    X_train_arr = X_train.to_numpy(dtype=float).copy()
    X_test_arr = X_test.to_numpy(dtype=float).copy()
    numeric_idx = [X.columns.get_loc(c) for c in NUMERIC_COLS]
    X_train_arr[:, numeric_idx] = scaler.fit_transform(
        X_train_arr[:, numeric_idx]
    )
    X_test_arr[:, numeric_idx] = scaler.transform(X_test_arr[:, numeric_idx])

    # Apply SMOTE to training set only
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train_arr, y_train.to_numpy()
    )
    # Fit final model on resampled data
    model = estimator_factory()
    model.fit(X_res, y_res)

    # Persist the held-out test set for bias_evaluation.py to use.
    test_df = pd.DataFrame(X_test_arr, columns=X.columns)
    test_df["Diabetes"] = y_test.to_numpy()
    test_df.to_csv(TEST_SET_PATH, index=False)

    return model, scaler


def main():
    # Load the features and target
    X, y = load_data()

    # Candidate models to compare
    candidates = {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        ),
    }

    print(f"Selecting best model by recall at threshold {THRESHOLD} "
          f"({N_SPLITS}-fold CV)\n")
    scores = {}
    # Run CV for each candidate
    for name, factory in candidates.items():
        print(f"  Evaluating {name}...")
        scores[name] = cv_recall_at_threshold(factory, X, y)
        s = scores[name]
        print(f"    recall {s['recall_mean']:.3f} ± {s['recall_std']:.3f}   "
              f"precision {s['precision_mean']:.3f} ± "
              f"{s['precision_std']:.3f}   AUC {s['auc_mean']:.3f}")

    # Pick the model with best mean recall
    best_name = max(scores, key=lambda n: scores[n]["recall_mean"])
    print(f"\nSelected: {best_name}")

    print("Refitting on 80/20 split and saving artefacts...")
    # Refit the winning model on train split
    model, scaler = fit_final_model(candidates[best_name], X, y)
    # Save model, scaler and feature list
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    # Also save a summary so the report can cite numbers.
    pd.DataFrame(scores).T.to_csv(
        os.path.join(BASE_DIR, "model_selection_summary.csv")
    )
    print(f"Saved model, scaler, feature list, test set, and summary.")


if __name__ == "__main__":
    main()
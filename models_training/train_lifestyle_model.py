
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "nhanes_cleaned_lifestyle.csv")
MODEL_PATH = os.path.join(BASE_DIR, "lifestyle_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "lifestyle_scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "lifestyle_imputer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "lifestyle_feature_columns.pkl")
TEST_SET_PATH = os.path.join(BASE_DIR, "lifestyle_test_set.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "lifestyle_selection_summary.csv")

THRESHOLD = 0.15
RANDOM_STATE = 42
N_SPLITS = 5

# Columns that should be scaled. Binary and categorical columns stay as-is.
NUMERIC_COLS = [
    "Age", "BMI", "WaistCM", "ActivityMinutes", "SedentaryHours",
    "AlcoholPerWeek", "SleepHours", "DietQuality", "MealsOutPerWeek",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Diabetes"]).copy()
    # One-hot ethnicity — same reasoning as clinical: NHANES codes are nominal.
    X = pd.get_dummies(X, columns=["Ethnicity"], drop_first=True)
    y = df["Diabetes"].astype(int)
    return X, y


def preprocess_fold(X_train, X_test, numeric_idx):
    """Impute then scale numeric columns. Fit on train only."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_imp[:, numeric_idx] = scaler.fit_transform(
        X_train_imp[:, numeric_idx]
    )
    X_test_imp[:, numeric_idx] = scaler.transform(X_test_imp[:, numeric_idx])
    return X_train_imp, X_test_imp, imputer, scaler


def cv_evaluate(estimator_factory, X, y):
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    numeric_idx = [X.columns.get_loc(c) for c in NUMERIC_COLS if c in X.columns]
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy()

    recalls, precisions, aucs = [], [], []
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_tr, X_te, _, _ = preprocess_fold(
            X_arr[train_idx].copy(), X_arr[test_idx].copy(), numeric_idx
        )
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
        model = estimator_factory()
        model.fit(X_res, y_res)
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        recalls.append(recall_score(y_te, preds))
        precisions.append(precision_score(y_te, preds, zero_division=0))
        aucs.append(roc_auc_score(y_te, probs))

    return {
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "auc_mean": float(np.mean(aucs)),
    }


def fit_final(estimator_factory, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    numeric_idx = [X.columns.get_loc(c) for c in NUMERIC_COLS if c in X.columns]
    X_train_arr = X_train.to_numpy(dtype=float)
    X_test_arr = X_test.to_numpy(dtype=float)

    X_train_proc, X_test_proc, imputer, scaler = preprocess_fold(
        X_train_arr.copy(), X_test_arr.copy(), numeric_idx
    )
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train_proc, y_train.to_numpy()
    )

    model = estimator_factory()
    model.fit(X_res, y_res)

    # Save held-out test set so bias_evaluation can load it without leakage.
    test_df = pd.DataFrame(X_test_proc, columns=X.columns)
    test_df["Diabetes"] = y_test.to_numpy()
    test_df.to_csv(TEST_SET_PATH, index=False)

    return model, scaler, imputer


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} rows, {X.shape[1]} features, "
          f"positive rate {y.mean():.2%}")

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

    print(f"\nSelecting by CV recall at threshold {THRESHOLD}:\n")
    scores = {}
    for name, factory in candidates.items():
        scores[name] = cv_evaluate(factory, X, y)
        s = scores[name]
        print(f"  {name:20s} recall {s['recall_mean']:.3f} ± "
              f"{s['recall_std']:.3f}   precision {s['precision_mean']:.3f}   "
              f"AUC {s['auc_mean']:.3f}")

    best_name = max(scores, key=lambda n: scores[n]["recall_mean"])
    print(f"\nSelected: {best_name}")

    model, scaler, imputer = fit_final(candidates[best_name], X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    pd.DataFrame(scores).T.to_csv(SUMMARY_PATH)
    print(f"\nSaved model, scaler, imputer, feature list, test set, summary.")


if __name__ == "__main__":
    main()
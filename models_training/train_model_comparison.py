import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nhanes_cleaned_clinical.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "model_comparison_results.csv")

FEATURES = ["SystolicBP", "DiastolicBP", "BMI", "Age", "Sex", "Ethnicity"]
TARGET_COL = "Diabetes"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "RIDAGEYR": "Age",
        "BMXBMI": "BMI",
        "BPXSY1": "SystolicBP",
        "BPXDI1": "DiastolicBP",
        "RIAGENDR": "Sex",
        "RIDRETH3": "Ethnicity",
        "Diabetes_binary": "Diabetes"
    })

    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test


def evaluate_model(name, model, X_train, X_test, y_train, y_test, threshold=0.15):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    y_pred_thresh = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)

    return {
        "model": name,
        "auc": auc,
        "accuracy_at_0.15": acc,
        "recall_at_0.15": rec
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("XGBoost", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )),
    ]

    results = []
    for name, clf in models:
        metrics = evaluate_model(name, clf, X_train, X_test, y_train, y_test)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved model comparison results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

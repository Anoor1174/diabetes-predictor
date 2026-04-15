import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("nhanes_cleaned_clinical.csv")

X = df[["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "BPXSY1", "BPXDI1"]]
y = df["Diabetes_binary"]

X["RIAGENDR"] = X["RIAGENDR"].map({1: 0, 2: 1})

X = pd.get_dummies(X, columns=["RIDRETH3"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_bal, y_train_bal)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print("Accuracy:", acc)
print("Recall:", rec)
print("Precision:", prec)

joblib.dump(model, "clinical_model.pkl")
joblib.dump(scaler, "clinical_scaler.pkl")

print("Model and scaler saved.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# 1. Load cleaned dataset
df = pd.read_csv("nhanes_cleaned_clinical.csv")

# 2. Feature selection
X = df[["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "BPXSY1", "BPXDI1"]]
y = df["Diabetes_binary"]

# 3. Encode categorical variables
# Sex: 1=Male, 2=Female → convert to 0/1
X["RIAGENDR"] = X["RIAGENDR"].map({1: 0, 2: 1})

# Ethnicity: one-hot encode
X = pd.get_dummies(X, columns=["RIDRETH3"], drop_first=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Balance dataset using SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

# 7. Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_bal, y_train_bal)

# 8. Evaluate at default threshold (0.5)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print("Accuracy:", acc)
print("Recall:", rec)
print("Precision:", prec)

# 9. Save model + scaler
joblib.dump(model, "clinical_model.pkl")
joblib.dump(scaler, "clinical_scaler.pkl")

print("Model and scaler saved.")

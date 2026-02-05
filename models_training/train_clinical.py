import os
from dataclasses import dataclass

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class DatasetConfig:
    path: str          #path to the dataset CSV
    target: str        #target column name
    features: list     #feature column names


class ClinicalModelTrainer:
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler = StandardScaler()

        #XGBoost configured for imbalanced screening data
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=40,   #handles class imbalance
            random_state=42
        )

    def load_data(self):
        #load up the NHANES dataset
        df = pd.read_csv(self.config.path)

        #fill missing values with median
        X = df[self.config.features].fillna(df[self.config.features].median())
        y = df[self.config.target]

        return X, y

    def split_and_balance(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        #here we apply SMOTE to balance the training data
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        return X_train_res, X_test, y_train_res, y_test

    def train(self):
        #load and prepare the data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.split_and_balance(X, y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        #trains the model
        self.model.fit(X_train_scaled, y_train)
        probs = self.model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs > 0.15).astype(int)

        print("\nClinical XGBoost Performance (threshold = 0.15):\n")
        print(classification_report(y_test, preds))

    def save(self):
        os.makedirs("app/models", exist_ok=True)

        #saves the trained model and scaler
        joblib.dump(self.model, "app/models/clinical_model.pkl")
        joblib.dump(self.scaler, "app/models/clinical_scaler.pkl")

        print("\nClinical model and scaler saved to app/models/")


if __name__ == "__main__":
    config = DatasetConfig(
        path="data/nhanes_diabetes.csv",
        target="Diabetes_binary",
        features=[
            "SystolicBP",
            "DiastolicBP",
            "BMI",
            "Age",
            "Sex",
            "Ethnicity"
        ]
    )

    #trains and save the model
    trainer = ClinicalModelTrainer(config)
    trainer.train()
    trainer.save()
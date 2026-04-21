import os
import joblib
import numpy as np

def test_model_loads_and_predicts():
    model_path = os.path.join("app", "models", "clinical_model.pkl")
    scaler_path = os.path.join("app", "models", "clinical_scaler.pkl")

    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    sample = np.array([[130, 80, 30, 45, 1, 3]])  # SBP, DBP, BMI, Age, Sex, Ethnicity
    sample_scaled = scaler.transform(sample)

    prob = model.predict_proba(sample_scaled)[0][1]

    assert 0 <= prob <= 1

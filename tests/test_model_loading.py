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

    # Scaler was fitted on 4 numeric features: Age, BMI, SystolicBP, DiastolicBP
    sample = np.array([[45, 28.5, 130, 80]])
    sample_scaled = scaler.transform(sample)

    assert sample_scaled.shape == (1, 4)
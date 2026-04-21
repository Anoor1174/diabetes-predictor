import json

def test_predict_clinical_valid(client):
    payload = {
        "Age": 45,
        "Sex": 1,
        "Ethnicity": 3,
        "BMI": 28.5,
        "SystolicBP": 130,
        "DiastolicBP": 82
    }

    response = client.post(
        "/api/predict_clinical",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "final_probability" in data
    assert 0 <= data["final_probability"] <= 1


def test_predict_clinical_invalid_missing_field(client):
    payload = {
        "Age": 45,
        # Missing Sex
        "Ethnicity": 3,
        "BMI": 28.5,
        "SystolicBP": 130,
        "DiastolicBP": 82
    }

    response = client.post(
        "/api/predict_clinical",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 400

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

#initialises flas
app = Flask(__name__)

#path to the folder where the trained model + scaler are stored
MODEL_DIR = os.path.join("app", "models")

#load the trained clinical model and its scaler into memory
clinical_model = joblib.load(os.path.join(MODEL_DIR, "clinical_model.pkl"))
clinical_scaler = joblib.load(os.path.join(MODEL_DIR, "clinical_scaler.pkl"))




def risk_category(prob):
    if prob < 0.15:
        return "Low"
    elif prob < 0.35:
        return "Medium"
    else:
        return "High"



def rule_based_adjustments(data, ml_prediction):
    rules_triggered = []


    if data["Age"] >= 70 and data["BMI"] >= 30:
        ml_prediction = 1
        rules_triggered.append("Age ≥ 70 and BMI ≥ 30")

    if data["SystolicBP"] >= 150:
        ml_prediction = 1
        rules_triggered.append("Systolic BP ≥ 150")

    if data["BMI"] >= 40:
        ml_prediction = 1
        rules_triggered.append("BMI ≥ 40")

    return ml_prediction, rules_triggered


@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical_api():
    data = request.json  #extract JSON sent from the frontend

    try:
        age = float(data.get("Age", 0))
        sex = float(data.get("Sex", 0))
        ethnicity = float(data.get("Ethnicity", 0))
        bmi = float(data.get("BMI", 0))
        sbp = float(data.get("SystolicBP", 0))
        dbp = float(data.get("DiastolicBP", 0))

        #basic clinical validation ranges
        if not (0 <= age <= 120):
            return jsonify({"error": "Age must be 0-120"}), 400
        if not (10 <= bmi <= 80):
            return jsonify({"error": "BMI must be 10-80"}), 400
        if not (80 <= sbp <= 200):
            return jsonify({"error": "SBP must be 80-200"}), 400
        if not (50 <= dbp <= 130):
            return jsonify({"error": "DBP must be 50-130"}), 400

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input data"}), 400


    features = np.array([
        data["SystolicBP"],
        data["DiastolicBP"],
        data["BMI"],
        data["Age"],
        data["Sex"],
        data["Ethnicity"]
    ]).reshape(1, -1)

    scaled = clinical_scaler.transform(features)

    #predicts probability of diabetes using the ML model
    ml_prob = clinical_model.predict_proba(scaled)[0][1]

    #convert probability into a binary prediction using threshold 0.15
    ml_pred = 1 if ml_prob > 0.15 else 0
    final_pred, rules = rule_based_adjustments(data, ml_pred)

    #combines ML probability with rule adjustments
    combined_score = ml_prob
    if len(rules) > 0:
        combined_score = min(1.0, ml_prob + 0.20)
    category = risk_category(combined_score)

    #return the prediction back to the frontend
    return jsonify({
        "ml_probability": float(ml_prob),
        "final_probability": float(combined_score),
        "prediction": int(final_pred),
        "risk_category": category,
        "rules_triggered": rules
    })



@app.route("/")
def home():
    return render_template("home.html")


@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")


@app.route("/advice")
def advice_page():
    return render_template("advice.html")


@app.route("/help")
def help_page():
    return render_template("help.html")



if __name__ == "__main__":
    app.run(debug=True)

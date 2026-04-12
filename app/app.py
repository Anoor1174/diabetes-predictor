from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# NEW IMPORTS FOR FAIRNESS + OPTIMISATION
from threshold_optimisation import evaluate_at_threshold, sweep_thresholds
from performance_fairness_comparison import compute_performance_fairness_comparison

# Initialise Flask
app = Flask(__name__)

# Path to model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load trained model + scaler
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
    data = request.json

    try:
        age = float(data.get("Age", 0))
        sex = float(data.get("Sex", 0))
        ethnicity = float(data.get("Ethnicity", 0))
        bmi = float(data.get("BMI", 0))
        sbp = float(data.get("SystolicBP", 0))
        dbp = float(data.get("DiastolicBP", 0))

        # Basic validation
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

    # Prepare features for model
    features = np.array([
        data["SystolicBP"],
        data["DiastolicBP"],
        data["BMI"],
        data["Age"],
        data["Sex"],
        data["Ethnicity"]
    ]).reshape(1, -1)

    scaled = clinical_scaler.transform(features)

    # ML probability
    ml_prob = clinical_model.predict_proba(scaled)[0][1]

    # Default threshold
    ml_pred = 1 if ml_prob > 0.15 else 0

    # Apply rule-based overrides
    final_pred, rules = rule_based_adjustments(data, ml_pred)

    # Combine ML + rules
    combined_score = ml_prob
    if len(rules) > 0:
        combined_score = min(1.0, ml_prob + 0.20)

    category = risk_category(combined_score)

    return jsonify({
        "ml_probability": float(ml_prob),
        "final_probability": float(combined_score),
        "prediction": int(final_pred),
        "risk_category": category,
        "rules_triggered": rules
    })



# 1. Metrics at a specific threshold
@app.route("/api/fairness_metrics", methods=["GET"])
def fairness_metrics():
    threshold = float(request.args.get("threshold", 0.5))
    results = evaluate_at_threshold(threshold)
    return jsonify(results)


# 2. Sweep thresholds (0.05 → 0.50)
@app.route("/api/threshold_sweep", methods=["GET"])
def threshold_sweep():
    results = sweep_thresholds()
    return jsonify(results)


# 3. comparison of performance and fairness
@app.route("/api/performance_fairness_comparison", methods=["GET"])
def performance_fairness_comparison_api():
    # Define thresholds to sweep (0.01 → 0.99)
    thresholds = [i/100 for i in range(1, 100)]

    # Compute performance vs fairness comparison
    all_points, frontier = compute_performance_fairness_comparison(thresholds)

    return jsonify({
        "all_points": all_points.to_dict(orient="records"),
        "performance_fairness_comparison": frontier.to_dict(orient="records")
    })


@app.route("/predict_clinical", methods=["POST"])
def predict_clinical():
    # 1. Get form values
    age = float(request.form.get("age"))
    bmi = float(request.form.get("bmi"))
    glucose = float(request.form.get("glucose"))
    # ... add the rest of your inputs

    # 2. Prepare input for model
    input_data = [[age, bmi, glucose]]  # example

    # 3. Load model + scaler
    model = pickle.load(open("clinical_model.pkl", "rb"))
    scaler = pickle.load(open("clinical_scaler.pkl", "rb"))

    scaled = scaler.transform(input_data)
    prob = model.predict_proba(scaled)[0][1]

    # 4. Convert probability → label
    if prob >= 0.5:
        label = "High Risk"
        explanation = "Your results indicate a high likelihood of Type 2 diabetes. Please consider contacting a GP for further testing."
    elif prob >= 0.25:
        label = "Moderate Risk"
        explanation = "Your results suggest a moderate risk. Lifestyle changes can significantly reduce your risk."
    else:
        label = "Low Risk"
        explanation = "Your results indicate a low risk. Continue maintaining healthy habits."

    # 5. Redirect to results page
    return redirect(url_for(
        "results_page",
        risk_score=round(prob, 2),
        risk_label=label,
        explanation=explanation
    ))


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")

@app.route("/lifestyle")
def lifestyle_page():
    return render_template("lifestyle.html")

@app.route("/advice")
def advice_page():
    return render_template("advice.html")

@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/diabetesinfo")
def diabetesinfo_page():
    return render_template("diabetes_information.html")

@app.route("/result")
def results_page():
    risk_score = request.args.get("risk_score", None)
    risk_label = request.args.get("risk_label", None)
    explanation = request.args.get("explanation", None)
    return render_template(
    "result.html",
    risk_score=risk_score,
    risk_label=risk_label,
    explanation=explanation
)


@app.route("/insights")
def insights_page():
    return render_template("insights.html")


if __name__ == "__main__":
    app.run(debug=True)

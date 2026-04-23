import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Import fairness evaluation helpers
from app.performance_fairness_comparison import (
    compute_performance_fairness_comparison,
)
from app.threshold_optimisation import evaluate_at_threshold, sweep_thresholds


# Create the Flask app instance
app = Flask(__name__)

# Location of saved model artefacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Screening threshold prioritises recall
DECISION_THRESHOLD = 0.15
# Upper bound for low-risk category
LOW_RISK_MAX = 0.15
# Upper bound for medium-risk category
MEDIUM_RISK_MAX = 0.35
# Probability boost when rules trigger
RULE_OVERRIDE_BOOST = 0.20

# Accepted NHANES ethnicity codes
VALID_ETHNICITY_CODES = {1, 2, 3, 4, 6, 7}
# Accepted sex codes
VALID_SEX_CODES = {0, 1}
# Accepted smoking status codes
VALID_SMOKING_CODES = {0, 1, 2}
# Accepted family history codes
VALID_FAMILY_HISTORY_CODES = {0, 1}

# Numeric columns scaled in lifestyle pipeline
LIFESTYLE_NUMERIC_COLS = [
    "Age", "BMI", "WaistCM", "ActivityMinutes", "SedentaryHours",
    "AlcoholPerWeek", "SleepHours", "DietQuality", "MealsOutPerWeek",
]


def _load_artefact(filename):
    # Load a saved model file from disk
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model artefact: {path}. "
            f"Run the relevant training script first."
        )
    return joblib.load(path)


# Load clinical pipeline artefacts at startup
clinical_model = _load_artefact("clinical_model.pkl")
clinical_scaler = _load_artefact("clinical_scaler.pkl")
clinical_features = _load_artefact("clinical_feature_columns.pkl")

# Load lifestyle pipeline artefacts at startup
lifestyle_model = _load_artefact("lifestyle_model.pkl")
lifestyle_scaler = _load_artefact("lifestyle_scaler.pkl")
lifestyle_imputer = _load_artefact("lifestyle_imputer.pkl")
lifestyle_features = _load_artefact("lifestyle_feature_columns.pkl")


def risk_category(prob):
    """Map a probability to a three-tier label."""
    # Low risk below the first cutoff
    if prob < LOW_RISK_MAX:
        return "Low"
    # Medium risk below the second cutoff
    if prob < MEDIUM_RISK_MAX:
        return "Medium"
    # Otherwise flag as high risk
    return "High"


def rule_based_adjustments(data):
    # Check extreme clinical values manually
    rules = []
    # Older adults with high BMI
    if data.get("Age", 0) >= 70 and data.get("BMI", 0) >= 30:
        rules.append("Age ≥ 70 and BMI ≥ 30")
    # Stage 2 hypertension threshold
    if data.get("SystolicBP", 0) >= 150:
        rules.append("Systolic BP ≥ 150")
    # Severe obesity threshold
    if data.get("BMI", 0) >= 40:
        rules.append("BMI ≥ 40")
    return rules


def risk_explanation(category, pathway):
    """Human-readable text for the result page."""
    # Low-risk message
    if category == "Low":
        base = (
            "Based on your responses, your estimated diabetes risk is low. "
            "Maintaining current habits is recommended."
        )
    # Medium-risk message
    elif category == "Medium":
        base = (
            "Your estimated risk is in the moderate range. Small changes "
            "to modifiable factors could reduce it further."
        )
    # High-risk message
    else:
        base = (
            "Your estimated risk is elevated. Consider speaking to a "
            "healthcare professional and reviewing modifiable factors "
            "such as activity level, diet, and weight."
        )
    # Screening disclaimer
    suffix = (
        " This screening tool does not replace a clinical diagnosis."
    )
    # Extra caveat for lifestyle pathway
    if pathway == "lifestyle":
        suffix = (
            " This lifestyle pathway does not use blood pressure data, "
            "so its accuracy is lower than the clinical pathway."
            + suffix
        )
    return base + suffix


def _validate_clinical_input(data):
    """Return (coerced_dict, error_message). error is None on success."""
    try:
        # Coerce raw values to numeric types
        coerced = {
            "Age": float(data.get("Age", 0)),
            "Sex": int(data.get("Sex", 0)),
            "Ethnicity": int(data.get("Ethnicity", 0)),
            "BMI": float(data.get("BMI", 0)),
            "SystolicBP": float(data.get("SystolicBP", 0)),
            "DiastolicBP": float(data.get("DiastolicBP", 0)),
        }
    except (ValueError, TypeError):
        # Reject non-numeric inputs
        return None, "Invalid input data: non-numeric field"

    # Range checks for each feature
    if not 18 <= coerced["Age"] <= 100:
        return None, "Age must be between 18 and 100"
    if coerced["Sex"] not in VALID_SEX_CODES:
        return None, "Sex must be 0 (male) or 1 (female)"
    if coerced["Ethnicity"] not in VALID_ETHNICITY_CODES:
        return None, "Ethnicity code not recognised"
    if not 13 <= coerced["BMI"] <= 70:
        return None, "BMI must be between 13 and 70"
    if not 80 <= coerced["SystolicBP"] <= 220:
        return None, "Systolic BP must be between 80 and 220"
    if not 40 <= coerced["DiastolicBP"] <= 130:
        return None, "Diastolic BP must be between 40 and 130"
    return coerced, None


def _build_clinical_row(data):
    """Build a single-row DataFrame aligned to the trained feature order."""
    # Wrap the input as a DataFrame
    row = pd.DataFrame([{
        "Age": data["Age"],
        "Sex": data["Sex"],
        "Ethnicity": data["Ethnicity"],
        "BMI": data["BMI"],
        "SystolicBP": data["SystolicBP"],
        "DiastolicBP": data["DiastolicBP"],
    }])
    # One-hot encode ethnicity
    row = pd.get_dummies(row, columns=["Ethnicity"], drop_first=True)
    # Align to saved training columns
    row = row.reindex(columns=clinical_features, fill_value=0)
    return row


@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical_api():
    # Read JSON payload from request
    data = request.get_json(silent=True) or {}
    # Validate and coerce the input
    coerced, error = _validate_clinical_input(data)
    if error:
        # Reject with 400 on bad input
        return jsonify({"error": error}), 400

    # Build feature row for the model
    row_df = _build_clinical_row(coerced)
    # Scale only the numeric columns
    numeric_cols = ["Age", "BMI", "SystolicBP", "DiastolicBP"]
    row_df[numeric_cols] = clinical_scaler.transform(row_df[numeric_cols])
    # Get raw model probability
    ml_prob = float(clinical_model.predict_proba(row_df)[0, 1])
    # Check clinical override rules
    rules = rule_based_adjustments(coerced)
    # Apply boost if any rule triggered
    final_prob = min(1.0, ml_prob + RULE_OVERRIDE_BOOST * bool(rules))
    # Binary prediction at the threshold
    prediction = int(final_prob >= DECISION_THRESHOLD)
    # Map probability to risk category
    category = risk_category(final_prob)

    # Return full prediction response
    return jsonify({
        "ml_probability": round(ml_prob, 3),
        "final_probability": round(final_prob, 3),
        "prediction": prediction,
        "risk_category": category,
        "rules_triggered": rules,
        "risk_score": round(final_prob, 3),
        "risk_label": f"{category} Clinical Risk",
        "explanation": risk_explanation(category, "clinical"),
        "pathway": "clinical",
    })


def _validate_lifestyle_input(data):
    """Return (coerced_dict, error_message)."""
    try:
        # Coerce lifestyle inputs to numeric types
        coerced = {
            "Age": float(data.get("Age", 0)),
            "Sex": int(data.get("Sex", 0)),
            "Ethnicity": int(data.get("Ethnicity", 0)),
            "BMI": float(data.get("BMI", 0)),
            # Waist is optional
            "WaistCM": (
                float(data["WaistCM"])
                if data.get("WaistCM") not in (None, "") else np.nan
            ),
            "ActivityMinutes": float(data.get("ActivityMinutes", 0)),
            "SedentaryHours": float(data.get("SedentaryHours", 0)),
            "SmokingStatus": int(data.get("SmokingStatus", 0)),
            "AlcoholPerWeek": float(data.get("AlcoholPerWeek", 0)),
            "SleepHours": float(data.get("SleepHours", 0)),
            "DietQuality": int(data.get("DietQuality", 3)),
            "MealsOutPerWeek": float(data.get("MealsOutPerWeek", 0)),
            "FamilyHistory": int(data.get("FamilyHistory", 0)),
        }
    except (ValueError, TypeError):
        # Reject non-numeric inputs
        return None, "Invalid input data: non-numeric field"

    # Range checks for each lifestyle feature
    if not 18 <= coerced["Age"] <= 100:
        return None, "Age must be between 18 and 100"
    if coerced["Sex"] not in VALID_SEX_CODES:
        return None, "Sex must be 0 (male) or 1 (female)"
    if coerced["Ethnicity"] not in VALID_ETHNICITY_CODES:
        return None, "Ethnicity code not recognised"
    if not 13 <= coerced["BMI"] <= 70:
        return None, "BMI must be between 13 and 70"
    if coerced["SmokingStatus"] not in VALID_SMOKING_CODES:
        return None, "Smoking status must be 0, 1 or 2"
    if coerced["FamilyHistory"] not in VALID_FAMILY_HISTORY_CODES:
        return None, "Family history must be 0 or 1"
    if not 1 <= coerced["DietQuality"] <= 5:
        return None, "Diet quality must be between 1 and 5"
    if not 0 <= coerced["ActivityMinutes"] <= 3000:
        return None, "Activity minutes must be between 0 and 3000"
    if not 0 <= coerced["SedentaryHours"] <= 24:
        return None, "Sedentary hours must be between 0 and 24"
    if not 0 <= coerced["SleepHours"] <= 14:
        return None, "Sleep hours must be between 0 and 14"
    return coerced, None


def _build_lifestyle_row(data):
    # Wrap the coerced input as a DataFrame
    row = pd.DataFrame([data])
    # One-hot encode ethnicity
    row = pd.get_dummies(row, columns=["Ethnicity"], drop_first=True)
    # Align to saved training columns
    row = row.reindex(columns=lifestyle_features, fill_value=0)
    # Impute missing values
    row_imputed = pd.DataFrame(
        lifestyle_imputer.transform(row),
        columns=lifestyle_features
    )
    # Scale only numeric columns by name
    numeric_cols = [c for c in LIFESTYLE_NUMERIC_COLS if c in lifestyle_features]
    row_imputed[numeric_cols] = lifestyle_scaler.transform(row_imputed[numeric_cols])
    return row_imputed


@app.route("/api/predict_lifestyle", methods=["POST"])
def predict_lifestyle_api():
    # Read JSON payload from request
    data = request.get_json(silent=True) or {}
    # Validate and coerce the input
    coerced, error = _validate_lifestyle_input(data)
    if error:
        # Reject with 400 on bad input
        return jsonify({"error": error}), 400

    # Build feature row for the model
    row_df = _build_lifestyle_row(coerced)
    # Get raw lifestyle model probability
    ml_prob = float(lifestyle_model.predict_proba(row_df)[0, 1])
    # Apply limited lifestyle rule overrides
    applicable_rules = []
    if coerced["Age"] >= 70 and coerced["BMI"] >= 30:
        applicable_rules.append("Age ≥ 70 and BMI ≥ 30")
    if coerced["BMI"] >= 40:
        applicable_rules.append("BMI ≥ 40")
    # Apply boost if any rule triggered
    final_prob = min(
        1.0, ml_prob + RULE_OVERRIDE_BOOST * bool(applicable_rules)
    )
    # Binary prediction at the threshold
    prediction = int(final_prob >= DECISION_THRESHOLD)
    # Map probability to risk category
    category = risk_category(final_prob)

    # Return full prediction response
    return jsonify({
        "ml_probability": round(ml_prob, 3),
        "final_probability": round(final_prob, 3),
        "prediction": prediction,
        "risk_category": category,
        "rules_triggered": applicable_rules,
        "risk_score": round(final_prob, 3),
        "risk_label": f"{category} Lifestyle Risk",
        "explanation": risk_explanation(category, "lifestyle"),
        "pathway": "lifestyle",
    })


@app.route("/api/pareto_frontier", methods=["GET"])
def pareto_frontier_api():
    """Alias for /api/performance_fairness_comparison to match dashboard.js naming."""
    # Sweep thresholds from 0.01 to 0.99
    thresholds = [i / 100 for i in range(1, 100)]
    # Compute all points and the frontier
    all_points, frontier = compute_performance_fairness_comparison(thresholds)
    return jsonify({
        "all_points": all_points.to_dict(orient="records"),
        "pareto_frontier": frontier.to_dict(orient="records"),
    })


@app.route("/api/fairness_metrics", methods=["GET"])
def fairness_metrics():
    # Read threshold from query string
    threshold = float(request.args.get("threshold", DECISION_THRESHOLD))
    # Return fairness metrics at that threshold
    return jsonify(evaluate_at_threshold(threshold))


@app.route("/api/threshold_sweep", methods=["GET"])
def threshold_sweep_api():
    # Sweep thresholds across the full range
    thresholds = [i / 100 for i in range(1, 100)]
    return jsonify(sweep_thresholds(thresholds))


@app.route("/api/performance_fairness_comparison", methods=["GET"])
def performance_fairness_comparison_api():
    # Sweep thresholds across the full range
    thresholds = [i / 100 for i in range(1, 100)]
    # Compute all points and the frontier
    all_points, frontier = compute_performance_fairness_comparison(thresholds)
    return jsonify({
        "all_points": all_points.to_dict(orient="records"),
        "performance_fairness_comparison": frontier.to_dict(orient="records"),
    })


# Home page route
@app.route("/")
def home():
    return render_template("home.html")


# Clinical form page
@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")


# Lifestyle form page
@app.route("/lifestyle")
def lifestyle_page():
    return render_template("lifestyle.html")


# Advice page
@app.route("/advice")
def advice_page():
    return render_template("advice.html")


# Help page
@app.route("/help")
def help_page():
    return render_template("help.html")


# Fairness dashboard page
@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


# Diabetes information page
@app.route("/diabetesinfo")
def diabetesinfo_page():
    return render_template("diabetes_information.html")


# Insights page
@app.route("/insights")
def insights_page():
    return render_template("insights.html")


# Result page reads query string
@app.route("/result")
def results_page():
    return render_template(
        "result.html",
        risk_score=request.args.get("risk_score"),
        risk_label=request.args.get("risk_label"),
        explanation=request.args.get("explanation"),
        pathway=request.args.get("pathway"),
    )


# Run the dev server
if __name__ == "__main__":
    app.run(debug=True)
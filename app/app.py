import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.performance_fairness_comparison import (
    compute_performance_fairness_comparison,
)
from app.threshold_optimisation import evaluate_at_threshold, sweep_thresholds


app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DECISION_THRESHOLD = 0.15
LOW_RISK_MAX = 0.15
MEDIUM_RISK_MAX = 0.35
RULE_OVERRIDE_BOOST = 0.20

VALID_ETHNICITY_CODES = {1, 2, 3, 4, 6, 7}
VALID_SEX_CODES = {0, 1}
VALID_SMOKING_CODES = {0, 1, 2}
VALID_FAMILY_HISTORY_CODES = {0, 1}

LIFESTYLE_NUMERIC_COLS = [
    "Age", "BMI", "WaistCM", "ActivityMinutes", "SedentaryHours",
    "AlcoholPerWeek", "SleepHours", "DietQuality", "MealsOutPerWeek",
]


def _load_artefact(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model artefact: {path}. "
            f"Run the relevant training script first."
        )
    return joblib.load(path)


clinical_model = _load_artefact("clinical_model.pkl")
clinical_scaler = _load_artefact("clinical_scaler.pkl")
clinical_features = _load_artefact("clinical_feature_columns.pkl")

lifestyle_model = _load_artefact("lifestyle_model.pkl")
lifestyle_scaler = _load_artefact("lifestyle_scaler.pkl")
lifestyle_imputer = _load_artefact("lifestyle_imputer.pkl")
lifestyle_features = _load_artefact("lifestyle_feature_columns.pkl")


def risk_category(prob):
    """Map a probability to a three-tier label."""
    if prob < LOW_RISK_MAX:
        return "Low"
    if prob < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def rule_based_adjustments(data):
    """Return a list of triggered clinical-rule flags.

    Rules are applied as *boosts* to the ML probability rather than
    hard overrides. This keeps the displayed probability, prediction,
    and category internally consistent — a rule firing adds
    RULE_OVERRIDE_BOOST to the probability, which then feeds into
    both the binary prediction and the risk category.
    """
    rules = []
    if data.get("Age", 0) >= 70 and data.get("BMI", 0) >= 30:
        rules.append("Age ≥ 70 and BMI ≥ 30")
    if data.get("SystolicBP", 0) >= 150:
        rules.append("Systolic BP ≥ 150")
    if data.get("BMI", 0) >= 40:
        rules.append("BMI ≥ 40")
    return rules


def risk_explanation(category, pathway):
    """Human-readable text for the result page."""
    if category == "Low":
        base = (
            "Based on your responses, your estimated diabetes risk is low. "
            "Maintaining current habits is recommended."
        )
    elif category == "Medium":
        base = (
            "Your estimated risk is in the moderate range. Small changes "
            "to modifiable factors could reduce it further."
        )
    else:
        base = (
            "Your estimated risk is elevated. Consider speaking to a "
            "healthcare professional and reviewing modifiable factors "
            "such as activity level, diet, and weight."
        )
    suffix = (
        " This screening tool does not replace a clinical diagnosis."
    )
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
        coerced = {
            "Age": float(data.get("Age", 0)),
            "Sex": int(data.get("Sex", 0)),
            "Ethnicity": int(data.get("Ethnicity", 0)),
            "BMI": float(data.get("BMI", 0)),
            "SystolicBP": float(data.get("SystolicBP", 0)),
            "DiastolicBP": float(data.get("DiastolicBP", 0)),
        }
    except (ValueError, TypeError):
        return None, "Invalid input data: non-numeric field"

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
    row = pd.DataFrame([{
        "Age": data["Age"],
        "Sex": data["Sex"],
        "Ethnicity": data["Ethnicity"],
        "BMI": data["BMI"],
        "SystolicBP": data["SystolicBP"],
        "DiastolicBP": data["DiastolicBP"],
    }])
    row = pd.get_dummies(row, columns=["Ethnicity"], drop_first=True)
    row = row.reindex(columns=clinical_features, fill_value=0)
    return row.to_numpy(dtype=float)


@app.route("/api/predict_clinical", methods=["POST"])
def predict_clinical_api():
    data = request.get_json(silent=True) or {}
    coerced, error = _validate_clinical_input(data)
    if error:
        return jsonify({"error": error}), 400

    arr = _build_clinical_row(coerced)
    scaled = clinical_scaler.transform(arr)
    ml_prob = float(clinical_model.predict_proba(scaled)[0, 1])

    rules = rule_based_adjustments(coerced)
    final_prob = min(1.0, ml_prob + RULE_OVERRIDE_BOOST * bool(rules))
    prediction = int(final_prob >= DECISION_THRESHOLD)
    category = risk_category(final_prob)

    return jsonify({
        "ml_probability": round(ml_prob, 3),
        "final_probability": round(final_prob, 3),
        "prediction": prediction,
        "risk_category": category,
        "rules_triggered": rules,
        "pathway": "clinical",
    })

def _validate_lifestyle_input(data):
    """Return (coerced_dict, error_message)."""
    try:
        coerced = {
            "Age": float(data.get("Age", 0)),
            "Sex": int(data.get("Sex", 0)),
            "Ethnicity": int(data.get("Ethnicity", 0)),
            "BMI": float(data.get("BMI", 0)),
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
        return None, "Invalid input data: non-numeric field"

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
    row = pd.DataFrame([data])
    row = pd.get_dummies(row, columns=["Ethnicity"], drop_first=True)
    row = row.reindex(columns=lifestyle_features, fill_value=0)
    arr = lifestyle_imputer.transform(row.to_numpy(dtype=float))
    numeric_idx = [
        lifestyle_features.index(c)
        for c in LIFESTYLE_NUMERIC_COLS if c in lifestyle_features
    ]
    arr[:, numeric_idx] = lifestyle_scaler.transform(arr[:, numeric_idx])
    return arr


@app.route("/api/predict_lifestyle", methods=["POST"])
def predict_lifestyle_api():
    data = request.get_json(silent=True) or {}
    coerced, error = _validate_lifestyle_input(data)
    if error:
        return jsonify({"error": error}), 400

    arr = _build_lifestyle_row(coerced)
    ml_prob = float(lifestyle_model.predict_proba(arr)[0, 1])

    # The lifestyle pathway has access to BMI but not BP, so only the
    # BMI-based rules apply. Systolic BP rule is skipped.
    applicable_rules = []
    if coerced["Age"] >= 70 and coerced["BMI"] >= 30:
        applicable_rules.append("Age ≥ 70 and BMI ≥ 30")
    if coerced["BMI"] >= 40:
        applicable_rules.append("BMI ≥ 40")

    final_prob = min(
        1.0, ml_prob + RULE_OVERRIDE_BOOST * bool(applicable_rules)
    )
    prediction = int(final_prob >= DECISION_THRESHOLD)
    category = risk_category(final_prob)

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
    thresholds = [i / 100 for i in range(1, 100)]
    all_points, frontier = compute_performance_fairness_comparison(thresholds)
    return jsonify({
        "all_points": all_points.to_dict(orient="records"),
        "pareto_frontier": frontier.to_dict(orient="records"),
    })


@app.route("/api/fairness_metrics", methods=["GET"])
def fairness_metrics():
    threshold = float(request.args.get("threshold", DECISION_THRESHOLD))
    return jsonify(evaluate_at_threshold(threshold))


@app.route("/api/threshold_sweep", methods=["GET"])
def threshold_sweep_api():
    thresholds = [i / 100 for i in range(1, 100)]
    return jsonify(sweep_thresholds(thresholds))


@app.route("/api/performance_fairness_comparison", methods=["GET"])
def performance_fairness_comparison_api():
    thresholds = [i / 100 for i in range(1, 100)]
    all_points, frontier = compute_performance_fairness_comparison(thresholds)
    return jsonify({
        "all_points": all_points.to_dict(orient="records"),
        "performance_fairness_comparison": frontier.to_dict(orient="records"),
    })


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


@app.route("/insights")
def insights_page():
    return render_template("insights.html")


@app.route("/result")
def results_page():
    return render_template(
        "result.html",
        risk_score=request.args.get("risk_score"),
        risk_label=request.args.get("risk_label"),
        explanation=request.args.get("explanation"),
        pathway=request.args.get("pathway"),
    )


if __name__ == "__main__":
    app.run(debug=True)
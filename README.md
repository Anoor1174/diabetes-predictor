# Dual-Pathway Diabetes Risk Predictor & Fairness Dashboard
### Final Year Project

An advanced, dual-pathway machine learning web application engineered for screening and predicting Type 2 diabetes risk. This system bridges the gap between patient-facing accessibility and clinical precision by offering two distinct risk-assessment engines alongside a native data-fairness audit dashboard.

---

## Key Structural Features

### 1. Dual-Pathway Assessment Architecture
* **Clinical Screening Pathway:** Engineered for high-accuracy assessment utilizing objective physiological metrics:
    * *Metrics:* Age, Sex, Ethnicity, Body Mass Index (BMI), Systolic Blood Pressure (BP), Diastolic Blood Pressure.
* **Lifestyle Screening Pathway:** A non-invasive screening entry point requiring no clinical equipment or metrics, enabling accessible preventative testing for users at home:
    * *Metrics:* Physical activity levels, nutritional/diet quality index, smoking status, alcohol consumption patterns, sleep duration, and hereditary family history.

### 2. Hybrid Decision Pipeline & Safety Overrides
To ensure absolute patient safety, the machine learning inference pipeline is wrapped with deterministic **Rule-Based Overrides**. The application bypasses model probability and immediately flags high-risk classification scenarios if critical clinical criteria are breached, such as:
* Critical obesity markers ($$\text{BMI} \ge 40$$)
* Stage 2 hypertensive crisis metrics ($$\text{Systolic BP} \ge 150\text{ mmHg}$$)

### 3. Demographic Bias & Algorithmic Fairness Dashboard
A dedicated auditing interface built to mitigate systemic healthcare discrepancies by tracking model equity across diverse population sectors:
* **Recall by Ethnicity Chart:** A dynamic visual comparison breaking down true positive recognition rates across different ethnic groups to detect under-diagnosis bias.
* **Pareto Frontier Scatter Plot:** A multi-objective evaluation engine graphing the explicit trade-off between absolute predictive recall vs algorithmic fairness constraints.


### 4. Gamified Kids' Health Module (Childhood Obesity & Diabetes Prevention) - Coming soon
An interactive, educational extension designed specifically to translate complex metabolic health concepts into an engaging, non-anxious experience for younger audiences:
* **The Daily Power Meter:** A simplified lifestyle logging interface where children track positive daily habits (hydration, nutrition, outdoor play) to visually fill an interactive energy bar.
* **Hero Quest Story Mode:** A choice-based text adventure path that guides kids through everyday scenarios, demonstrating how balanced choices defeat low-energy obstacles.
* **Achievement Badges:** A reward mechanic that awards digital badges (such as Hydration Hero or Step Master) for maintaining healthy consistency.
* **Myth-Busting Trivia Battles:** A rapid-fire interactive quiz module designed to challenge common misconceptions about food and physical health using instant feedback.
---

## Technical Infrastructure Stack

* **Core Backend Architecture:** Python, Flask (Inference Routing & Page Templates)
* **Machine Learning Engine:** scikit-learn, XGBoost (Gradient Boosting Classifiers)
* **Imbalanced Data Engineering:** imbalanced-learn (SMOTE / Synthetic Minority Over-sampling Technique)
* **Data Pipelines & Sanitization:** Pandas, NumPy
* **User Interface Layer:** Semantic HTML5, CSS3 Custom Properties, Modern JavaScript (ES6+), Jinja2 Templating Engine
* **Data Visualization Engine:** Chart.js (Real-time HTML5 Canvas rendering)
* **Quality Assurance & Verification:** Pytest (Unit Testing Framework)

---

## Project Structure & Architecture

```text
diabetes-predictor/
├── app/
│   ├── models/             # Serialized ML pipeline files (.pkl / .json)
│   ├── static/             # CSS styling variables, JS engine, charts logic
│   │   ├── css/            # Dark clinical-terminal layout styles
│   │   └── js/             # Evaluation handlers & Chart.js instances
│   ├── templates/          # Jinja2 HTML web interface panels
│   ├── routes.py           # API processing endpoints & clinical routing
│   └── utils.py            # Rule-based safety overrides & fairness math
├── tests/                  # Automated Pytest suite modules
├── requirements.txt        # System library dependencies manifests
├── run.py                  # Local environment application runtime execution hook
└── README.md               # Primary project documentation asset

```

---

## Installation & Workspace Setup

**[Launch Live Diabetes Risk Predictor Console](https://diabetes-predictor-81py.onrender.com/dashboard)**


# Diabetes Risk Predictor
A dual-pathway machine learning web application for Type 2 Diabetes risk screening


## Key Features:
- Clinical pathway: Risk prediction using physiological measurements
- Age, Sex, Ethnicity, BMI, Systolic BP, Diastolic BP
- Lifestyle pathway: Non-invasive risk prediction for users without clinical data
- Activity level, diet quality, smoking, alcohol, sleep, family history
- Rule-based overrides: Flags clear high-risk cases such as BMI ≥ 40 or systolic BP ≥ 150
- Fairness dashboard: Visualises model performance across ethnic groups
- Recall by ethnicity bar chart
- Pareto frontier scatter plot for the recall vs fairness trade-off
- NHS inspired UI:  Designed for familiarity and trust using NHS identity guidelines


## Tech Stack
- Backend: Python, Flask
- Machine Learning: Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- Data Handling: Pandas, NumPy
- Frontend: HTML, CSS, JavaScript, Jinja2
- Visualisation: Chart.js
- Testing: Pytest
- Version Control: Git, GitHub


## Installation
1. Clone the repository
bashgit clone https://github.com/Anoor1174/diabetes-predictor.git
2. run the backend using python run.py
3. open command promnt and navigate into cd diabetes-predictor
4. The application runs at http://127.0.0.1:5000/






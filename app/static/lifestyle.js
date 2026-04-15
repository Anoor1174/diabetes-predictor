// lifestyle.js
// Computes lifestyle risk score and redirects to /result

function submitLifestyle() {
    // Read inputs safely
    const activity = Number(document.getElementById("activity_minutes").value || 0);
    const sedentary = Number(document.getElementById("sedentary_hours").value || 0);
    const smoking = document.getElementById("smoking_status").value;
    const fruitVeg = Number(document.getElementById("fruit_veg").value || 0);
    const sugary = Number(document.getElementById("sugary_drinks").value || 0);
    const fastFood = Number(document.getElementById("fast_food").value || 0);
    const alcohol = Number(document.getElementById("alcohol").value || 0);
    const sleep = Number(document.getElementById("sleep_hours").value || 0);

    // Lifestyle scoring (simple, explainable, clinically aligned)
    let score = 0;

    // Physical activity: less activity → higher risk
    score += Math.max(0, 150 - activity) / 150;

    // Sedentary time
    score += Math.min(sedentary / 10, 1);

    // Smoking
    if (smoking === "current") score += 1;
    if (smoking === "former") score += 0.5;

    // Diet
    score += Math.max(0, 5 - fruitVeg) / 5;
    score += Math.min((sugary + fastFood) / 10, 1);

    // Alcohol
    score += Math.min(alcohol / 14, 1);

    // Sleep (ideal = 7–9 hours)
    score += Math.abs(8 - sleep) / 8;

    // Normalise to 0–1
    score = Math.min(score / 6, 1);
    const riskScore = Number(score.toFixed(2));

    // Risk label + explanation
    let label, explanation;

    if (riskScore < 0.33) {
        label = "Low Lifestyle Risk";
        explanation = "Your lifestyle pattern suggests a relatively low contribution to diabetes risk.";
    } else if (riskScore < 0.66) {
        label = "Moderate Lifestyle Risk";
        explanation = "Some lifestyle factors may be increasing your risk. Small changes could make a big difference.";
    } else {
        label = "High Lifestyle Risk";
        explanation = "Several lifestyle factors are likely increasing your diabetes risk. Consider reviewing your habits.";
    }

    // Redirect to results page (same as clinical path)
    window.location.href =
        `/result?risk_score=${riskScore}&risk_label=${encodeURIComponent(label)}&explanation=${encodeURIComponent(explanation)}`;
}

// Handle lifestyle form submission
async function submitLifestyle(event) {
    // Stop the browser default submit
    event.preventDefault();
    // Hide any previous error messages
    const errorBox = document.getElementById("error");
    errorBox.style.display = "none";

    // Read height and weight for BMI
    const heightCm = Number(document.getElementById("height_cm").value);
    const weightKg = Number(document.getElementById("weight_kg").value);
    // Calculate BMI from height and weight
    const bmi = weightKg / Math.pow(heightCm / 100, 2);

    // Build the payload for the API
    const payload = {
        Age: Number(document.getElementById("age").value),
        Sex: Number(document.getElementById("sex").value),
        Ethnicity: Number(document.getElementById("ethnicity").value),
        BMI: Number(bmi.toFixed(1)),
        // Waist is optional
        WaistCM: document.getElementById("waist_cm").value
            ? Number(document.getElementById("waist_cm").value) : null,
        ActivityMinutes: Number(document.getElementById("activity_minutes").value),
        SedentaryHours: Number(document.getElementById("sedentary_hours").value),
        SmokingStatus: Number(document.getElementById("smoking_status").value),
        AlcoholPerWeek: Number(document.getElementById("alcohol").value),
        SleepHours: Number(document.getElementById("sleep_hours").value),
        DietQuality: Number(document.getElementById("diet_quality").value),
        MealsOutPerWeek: Number(document.getElementById("meals_out").value),
        FamilyHistory: Number(document.getElementById("family_history").value),
    };

    try {
        // Send prediction request to Flask API
        const response = await fetch("/api/predict_lifestyle", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        // Throw error if response not OK
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        // Parse JSON response body
        const data = await response.json();

        // Redirect to results page with prediction data
        const params = new URLSearchParams({
            risk_score: data.risk_score,
            risk_label: data.risk_label,
            explanation: data.explanation,
            pathway: "lifestyle",
        });
        window.location.href = `/result?${params.toString()}`;
    } catch (err) {
        // Display error message to user
        errorBox.textContent = `Could not calculate risk: ${err.message}`;
        errorBox.style.display = "block";
    }
}
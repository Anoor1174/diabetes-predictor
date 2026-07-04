// Handle lifestyle form submission
function optionalNumber(id) {
    // Returns null for a blank/untouched field so the backend imputer
    // fills it with the population median instead of assuming zero.
    const el = document.getElementById(id);
    return el.value === "" ? null : Number(el.value);
}

async function submitLifestyle(event) {
    // Stop the browser default submit
    event.preventDefault();
    // Hide any previous error messages
    const errorBox = document.getElementById("error");
    errorBox.style.display = "none";

    // Required fields
    const age = document.getElementById("age").value;
    const sex = document.getElementById("sex").value;
    const ethnicity = document.getElementById("ethnicity").value;
    const heightCm = document.getElementById("height_cm").value;
    const weightKg = document.getElementById("weight_kg").value;
    const familyHistory = document.getElementById("family_history").value;

    if (!age || sex === "" || ethnicity === "" || !heightCm || !weightKg || familyHistory === "") {
        errorBox.textContent = "Please fill in the essentials section before calculating your risk.";
        errorBox.style.display = "block";
        return;
    }

    // Calculate BMI from height and weight
    const bmi = Number(weightKg) / Math.pow(Number(heightCm) / 100, 2);

    // Build the payload for the API. Optional fields fall back to null,
    // which the backend imputer fills rather than treating as zero.
    const payload = {
        Age: Number(age),
        Sex: Number(sex),
        Ethnicity: Number(ethnicity),
        BMI: Number(bmi.toFixed(1)),
        WaistCM: optionalNumber("waist_cm"),
        ActivityMinutes: optionalNumber("activity_minutes"),
        SedentaryHours: optionalNumber("sedentary_hours"),
        SmokingStatus: optionalNumber("smoking_status"),
        AlcoholPerWeek: optionalNumber("alcohol"),
        SleepHours: optionalNumber("sleep_hours"),
        DietQuality: optionalNumber("diet_quality"),
        MealsOutPerWeek: optionalNumber("meals_out"),
        FamilyHistory: Number(familyHistory),
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

async function submitLifestyle(event) {
    event.preventDefault();
    const errorBox = document.getElementById("error");
    errorBox.style.display = "none";

    const heightCm = Number(document.getElementById("height_cm").value);
    const weightKg = Number(document.getElementById("weight_kg").value);
    const bmi = weightKg / Math.pow(heightCm / 100, 2);

    const payload = {
        Age: Number(document.getElementById("age").value),
        Sex: Number(document.getElementById("sex").value),
        Ethnicity: Number(document.getElementById("ethnicity").value),
        BMI: Number(bmi.toFixed(1)),
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
        const response = await fetch("/api/predict_lifestyle", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const data = await response.json();

        const params = new URLSearchParams({
            risk_score: data.risk_score,
            risk_label: data.risk_label,
            explanation: data.explanation,
            pathway: "lifestyle",
        });
        window.location.href = `/result?${params.toString()}`;
    } catch (err) {
        errorBox.textContent = `Could not calculate risk: ${err.message}`;
        errorBox.style.display = "block";
    }
}
document.addEventListener("DOMContentLoaded", () => {

    document.getElementById("predict-btn").addEventListener("click", async () => {

        // Retrieves the user inputs
        const age = document.getElementById("age").value;
        const sex = document.getElementById("sex").value;
        const ethnicity = document.getElementById("ethnicity").value;
        const bmi = document.getElementById("bmi").value;
        const sbp = document.getElementById("sbp").value;
        const dbp = document.getElementById("dbp").value;

        // Mandatory field check
        if (!age || !sex || !ethnicity || !bmi || !sbp || !dbp) {
            alert("Please fill in all fields before estimating your risk.");
            return;
        }

        // Prepares the data to send to the backend API
        const payload = {
            Age: parseFloat(age),
            Sex: parseFloat(sex),
            Ethnicity: parseFloat(ethnicity),
            BMI: parseFloat(bmi),
            SystolicBP: parseFloat(sbp),
            DiastolicBP: parseFloat(dbp)
        };

        const resultBox = document.getElementById("clinical-result");
        resultBox.innerHTML = "Calculating...";

        try {
            const response = await fetch("/api/predict_clinical", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`Server returned ${response.status}`);

            const result = await response.json();

            // Redirect to results page with the prediction data
            const params = new URLSearchParams({
                risk_score: result.risk_score,
                risk_label: result.risk_label,
                explanation: result.explanation,
                pathway: "clinical",
            });
            window.location.href = `/result?${params.toString()}`;

        } catch (err) {
            resultBox.innerHTML = `<p style="color:red;">Could not calculate risk: ${err.message}</p>`;
        }
    });

});
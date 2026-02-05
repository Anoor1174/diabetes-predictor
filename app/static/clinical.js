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

        const response = await fetch("/api/predict_clinical", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        let colour = "#007f3b"; 
        if (result.risk_category === "Medium") colour = "#ffbf00";
        if (result.risk_category === "High") colour = "#d4351c";

        const barWidth = Math.min(100, result.final_probability * 100);

        resultBox.innerHTML = `
            <h3 style="color:${colour}">Risk Category: ${result.risk_category}</h3>

            <p><strong>Final Probability:</strong> ${(result.final_probability * 100).toFixed(1)}%</p>

            <div style="background:#ddd; width:100%; height:14px; border-radius:6px;">
                <div style="width:${barWidth}%; height:14px; background:${colour}; border-radius:6px;"></div>
            </div>

            <p style="margin-top:15px;"><strong>ML Probability:</strong> ${(result.ml_probability * 100).toFixed(1)}%</p>

        `;
    });

});

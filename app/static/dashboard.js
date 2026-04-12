const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");

let recallChart;
let paretoChart;

async function loadMetrics(threshold) {
    try {
        const res = await fetch(`/api/fairness_metrics?threshold=${threshold}`);
        const data = await res.json();

        // Update text metrics
        document.getElementById("overallAccuracy").innerText =
            data.overall_accuracy.toFixed(3);

        document.getElementById("overallRecall").innerText =
            data.overall_recall.toFixed(3);

        document.getElementById("fairnessGap").innerText =
            data.fairness_gap_recall.toFixed(3);

        // Prepare recall-by-ethnicity chart
        const labels = Object.keys(data.groups);
        const recalls = labels.map(k => data.groups[k].recall);

        if (recallChart) recallChart.destroy();

        recallChart = new Chart(document.getElementById("recallChart"), {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Recall",
                    data: recalls,
                    backgroundColor: "#3b82f6"
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

    } catch (err) {
        console.error("Error loading metrics:", err);
    }
}

async function loadPareto() {
    try {
        const res = await fetch("/api/pareto_frontier");
        const data = await res.json();

        const all = data.all_points;
        const frontier = data.pareto_frontier;

        if (paretoChart) paretoChart.destroy();

        paretoChart = new Chart(document.getElementById("paretoChart"), {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "All Thresholds",
                        data: all.map(p => ({
                            x: p.fairness_gap_recall,
                            y: p.overall_accuracy
                        })),
                        backgroundColor: "rgba(0,0,0,0.2)"
                    },
                    {
                        label: "Pareto Frontier",
                        data: frontier.map(p => ({
                            x: p.fairness_gap_recall,
                            y: p.overall_accuracy
                        })),
                        backgroundColor: "#ef4444",
                        pointRadius: 6
                    }
                ]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: "Fairness Gap (Recall)" }},
                    y: { title: { display: true, text: "Accuracy" }}
                }
            }
        });

    } catch (err) {
        console.error("Error loading Pareto frontier:", err);
    }
}

thresholdSlider.addEventListener("input", () => {
    const t = parseFloat(thresholdSlider.value).toFixed(2);
    thresholdValue.innerText = t;
    loadMetrics(t);
});


loadMetrics(0.5);
loadPareto();

import pandas as pd
import numpy as np

def compute_pareto_frontier(results):
    """
    Given a list of threshold evaluation results,
    compute the Pareto frontier for:
    - Minimising fairness gap
    - Maximising accuracy
    """

    # Convert to DataFrame for easier processing
    df = pd.DataFrame([
        {
            "threshold": r["threshold"],
            "accuracy": r["overall_accuracy"],
            "fairness_gap": r["fairness_gap_recall"]
        }
        for r in results
    ])

    # Sort by fairness gap (ascending)
    df = df.sort_values("fairness_gap")

    pareto_points = []
    best_accuracy_so_far = -1

    for _, row in df.iterrows():
        if row["accuracy"] > best_accuracy_so_far:
            pareto_points.append(row)
            best_accuracy_so_far = row["accuracy"]

    pareto_df = pd.DataFrame(pareto_points)

    return df, pareto_df
def compute_pareto_frontier(thresholds):
    from threshold_optimisation import sweep_thresholds  # lazy import to avoid circular dependency

    results = sweep_thresholds(thresholds)

    # your pareto logic here
    frontier = []
    for r in results:
        frontier.append({
            "threshold": r["threshold"],
            "overall_recall": r["overall_recall"],
            "fairness_gap_recall": r["fairness_gap_recall"]
        })

    return frontier


if __name__ == "__main__":
    print("Sweeping thresholds...")
    results = sweep_thresholds()

    print("Computing Pareto frontier...")
    all_points, pareto_frontier = compute_pareto_frontier(results)

    print("All points:", all_points.head())
    print("Pareto frontier:", pareto_frontier)

    # Save for dashboard
    all_points.to_csv("threshold_tradeoff_points.csv", index=False)
    pareto_frontier.to_csv("pareto_frontier.csv", index=False)

    print("Saved threshold_tradeoff_points.csv and pareto_frontier.csv")

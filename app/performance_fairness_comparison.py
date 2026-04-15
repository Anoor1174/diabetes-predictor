import pandas as pd

def compute_performance_fairness_comparison(thresholds):
    """
    Runs threshold sweep and computes performance vs fairness comparison.
    Returns:
        - all_points: DataFrame of all threshold results
        - frontier: DataFrame of Pareto-optimal points
    """

    from threshold_optimisation import sweep_thresholds  # avoid circular import

    # Run threshold sweep
    results = sweep_thresholds(thresholds)

    # Convert results to DataFrame
    df = pd.DataFrame([
        {
            "threshold": r["threshold"],
            "overall_accuracy": r["overall_accuracy"],
            "overall_recall": r["overall_recall"],
            "fairness_gap_recall": r["fairness_gap_recall"]
        }
        for r in results
    ])

    # Sort by fairness gap (ascending)
    df_sorted = df.sort_values("fairness_gap_recall")

    # Build Pareto frontier (min fairness gap, max recall)
    frontier_points = []
    best_recall_so_far = -1

    for _, row in df_sorted.iterrows():
        if row["overall_recall"] > best_recall_so_far:
            frontier_points.append(row)
            best_recall_so_far = row["overall_recall"]

    frontier_df = pd.DataFrame(frontier_points)

    return df, frontier_df


# Optional manual test
if __name__ == "__main__":
    from app.threshold_optimisation import sweep_thresholds

    print("Sweeping thresholds...")
    thresholds = [i/100 for i in range(1, 100)]
    results = sweep_thresholds(thresholds)

    print("Computing performance vs fairness comparison...")
    all_points, frontier = compute_performance_fairness_comparison(thresholds)

    print(all_points.head())
    print(frontier.head())

    all_points.to_csv("threshold_tradeoff_points.csv", index=False)
    frontier.to_csv("pareto_frontier.csv", index=False)

    print("Saved CSV files.")

import pandas as pd

from app.threshold_optimisation import sweep_thresholds



def compute_performance_fairness_comparison(thresholds):
    results = sweep_thresholds(thresholds)
    if not results:
        empty = pd.DataFrame(
            columns=["threshold", "overall_accuracy",
                     "overall_recall", "fairness_gap_recall"]
        )
        return empty, empty.copy()

    df = pd.DataFrame([
        {
            "threshold": r["threshold"],
            "overall_accuracy": r["overall_accuracy"],
            "overall_recall": r["overall_recall"],
            "fairness_gap_recall": r["fairness_gap_recall"],
        }
        for r in results
    ]).dropna(subset=["overall_recall", "fairness_gap_recall"])

    frontier_mask = []
    for i, row in df.iterrows():
        dominated = (
            (df["fairness_gap_recall"] <= row["fairness_gap_recall"])
            & (df["overall_recall"] >= row["overall_recall"])
            & (
                (df["fairness_gap_recall"] < row["fairness_gap_recall"])
                | (df["overall_recall"] > row["overall_recall"])
            )
        )
        dominated.loc[i] = False  # a point doesn't dominate itself
        frontier_mask.append(not dominated.any())

    frontier = (
        df[frontier_mask]
        .sort_values("fairness_gap_recall")
        .reset_index(drop=True)
    )
    return df.reset_index(drop=True), frontier


if __name__ == "__main__":
    thresholds = [i / 100 for i in range(1, 100)]
    print("Sweeping thresholds and building Pareto frontier...")
    all_points, frontier = compute_performance_fairness_comparison(thresholds)
    print(f"Evaluated {len(all_points)} thresholds; "
          f"{len(frontier)} on the frontier.")
    all_points.to_csv("threshold_tradeoff_points.csv", index=False)
    frontier.to_csv("pareto_frontier.csv", index=False)
    print("Saved CSVs.")
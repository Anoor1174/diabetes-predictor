import pandas as pd
import numpy as np
from app.threshold_optimisation import sweep_thresholds


def compute_performance_fairness_comparison(thresholds):
    """
    Runs threshold sweep and computes performance vs fairness comparison.
    
    Returns:
        - all_points: DataFrame of all threshold results
        - frontier: DataFrame of Pareto-optimal points
    """
    # Run the sweep across all thresholds
    results = sweep_thresholds(thresholds)
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Columns required for the comparison
    required_cols = [
        "threshold",
        "overall_accuracy",
        "overall_recall",
        "fairness_gap_recall",
    ]
    # Validate expected columns exist
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column in sweep results: {col}")

    # Pareto frontier across three metrics:
    # We want HIGH recall, HIGH accuracy, LOW fairness gap.
    # A point is dominated if any other point is at least as good on all three
    # and strictly better on at least one.
    frontier_mask = []
    # Check each candidate point
    for i, row in df.iterrows():
        dominated = False
        # Compare against every other point
        for j, other in df.iterrows():
            if j == i:
                continue
            # Check if the other point is at least as good everywhere
            at_least_as_good = (
                other["overall_recall"] >= row["overall_recall"]
                and other["overall_accuracy"] >= row["overall_accuracy"]
                and other["fairness_gap_recall"] <= row["fairness_gap_recall"]
            )
            # And strictly better on at least one metric
            strictly_better_somewhere = (
                other["overall_recall"] > row["overall_recall"]
                or other["overall_accuracy"] > row["overall_accuracy"]
                or other["fairness_gap_recall"] < row["fairness_gap_recall"]
            )
            # Mark as dominated if both conditions hold
            if at_least_as_good and strictly_better_somewhere:
                dominated = True
                break
        # Keep the point if it was never dominated
        frontier_mask.append(not dominated)

    # Filter to only non-dominated points
    frontier = df[frontier_mask].reset_index(drop=True)
    return df, frontier
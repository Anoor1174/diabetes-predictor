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
    results = sweep_thresholds(thresholds)
    df = pd.DataFrame(results)

    required_cols = [
        "threshold",
        "overall_accuracy",
        "overall_recall",
        "fairness_gap_recall",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column in sweep results: {col}")

    # Pareto frontier across three metrics:
    # We want HIGH recall, HIGH accuracy, LOW fairness gap.
    # A point is dominated if any other point is at least as good on all three
    # and strictly better on at least one.
    frontier_mask = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if j == i:
                continue
            at_least_as_good = (
                other["overall_recall"] >= row["overall_recall"]
                and other["overall_accuracy"] >= row["overall_accuracy"]
                and other["fairness_gap_recall"] <= row["fairness_gap_recall"]
            )
            strictly_better_somewhere = (
                other["overall_recall"] > row["overall_recall"]
                or other["overall_accuracy"] > row["overall_accuracy"]
                or other["fairness_gap_recall"] < row["fairness_gap_recall"]
            )
            if at_least_as_good and strictly_better_somewhere:
                dominated = True
                break
        frontier_mask.append(not dominated)

    frontier = df[frontier_mask].reset_index(drop=True)
    return df, frontier

# src/hpo/score.py
from __future__ import annotations
from typing import List
import numpy as np

def composite_score(map50: float, rmse: float, gt_counts: List[int]) -> float:
    mean_gt = float(np.mean(gt_counts)) if len(gt_counts) else 1.0
    rmse_norm = rmse / max(1.0, mean_gt)

    return 0.7 * float(map50 or 0.0) - 0.3 * float(rmse_norm)


def fitness(x, counting_metrics=None, optimize_for_counting=False):
    """
    Modified to integrate RMSE/MAE for counting tasks.
    Combines standard detection metrics with object counting metrics.
    """
    # If no counting metrics are provided, fall back to default YOLOv5 fitness
    if counting_metrics is None:
        w = [0.0, 0.0, 0.1, 0.9]  # Default weights
        return (x[:, :4] * w).sum(1)
        
    # Extract counting metrics
    rmse = counting_metrics.get('rmse', float('inf'))
    mae = counting_metrics.get('mae', float('inf'))

    # Normalize counting metrics (lower is better -> higher is better score)
    rmse_score = 1.0 / (1.0 + rmse)
    mae_score = 1.0 / (1.0 + mae)

    # Get the mAP@0.5 score from the results array 'x'
    map50 = x[:, 2]

    # Weighted combination of metrics
    if optimize_for_counting:
        # Prioritize counting metrics
        fi = (
            0.1 * map50 +           # 10% weight on mAP@0.5
            0.45 * rmse_score +     # 45% weight on RMSE
            0.45 * mae_score        # 45% weight on MAE
        )
    else:
        # Balanced approach
        fi = (
            0.5 * map50 +           # 50% weight on mAP@0.5
            0.25 * rmse_score +     # 25% weight on RMSE
            0.25 * mae_score        # 25% weight on MAE
        )
    return fi

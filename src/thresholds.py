import numpy as np
from typing import List, Tuple
from .metrics import compute_metrics

def find_best_threshold(
    all_scores: List[np.ndarray], 
    all_y_true: List[np.ndarray],
    resolution_minutes: int = 5
) -> Tuple[float, float]:
    """
    Finds the best anomaly threshold by minimizing a cost function.
    The metrics (DR, FAR, MTTD) for the cost function are calculated globally
    across all streams, according to formal definitions.
    """
    pooled_scores = np.concatenate(all_scores)

    # Generate a range of candidate thresholds from the anomaly scores
    tau_candidates = np.percentile(pooled_scores, np.linspace(80, 100, 100))

    best_tau = tau_candidates[0]
    best_cost = float('inf')

    # For each candidate threshold, calculate the global metrics and cost
    for tau in tau_candidates:
        
        # --- Accumulate counts and values across all streams ---
        total_correctly_detected = 0  # CDC = Correctly Detected Congestions
        total_incidents = 0             # NC = Number Of Congestions
        total_false_alarms = 0          # Sum of False Positives (FP)
        total_negatives = 0             # Sum of Negatives (FP + TN)
        sum_of_detection_times_minutes = 0.0

        for i in range(len(all_scores)):
            # Check if the stream actually contains an incident
            if np.sum(all_y_true[i]) > 0:
                total_incidents += 1
            
            metrics = compute_metrics(all_y_true[i], all_scores[i], tau, resolution_minutes)
            
            total_false_alarms += metrics['false_alarms']
            total_negatives += metrics['total_negatives']
            
            if metrics['detected']:
                total_correctly_detected += 1
                # Convert MTTD from hours back to minutes for the cost function
                sum_of_detection_times_minutes += (metrics['mttd_hours'] * 60.0)

        # --- Calculate final metrics based on formal definitions ---
        
        # DR = CDC / NC
        dr = total_correctly_detected / total_incidents if total_incidents > 0 else 1.0
        
        # FAR = sum(FP) / sum(FP + TN)
        far = total_false_alarms / total_negatives if total_negatives > 0 else 0.0
        
        # MTTD = sum(DT - AT) / CDC
        mttd_minutes = sum_of_detection_times_minutes / total_correctly_detected if total_correctly_detected > 0 else 0.0

        # --- Calculate the final cost ---
        # The cost function is designed to penalize low DR and high FAR, with a minor penalty for MTTD
        final_cost = (100 * (1 - dr)) + (100 * far) + mttd_minutes

        # Update the best threshold if the current cost is lower
        if final_cost < best_cost:
            best_cost = final_cost
            best_tau = tau
            
    return best_tau, best_cost
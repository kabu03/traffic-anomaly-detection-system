import numpy as np

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, tau: float, resolution_minutes: int = 5):
    """
    Calculates key time-series anomaly detection metrics for a single stream.
    This function assumes it is given ONLY the data for the evaluation period.
    Returns raw counts for aggregation, not final ratios.
    """
    y_pred = (scores > tau).astype(int)
    
    # --- False Alarm Counts ---
    # The arrays y_true and y_pred are now assumed to be the holdout/test set.
    actual_negatives_mask = (y_true == 0)
    
    false_alarms = int(np.sum(y_pred[actual_negatives_mask]))
    total_negatives = int(np.sum(actual_negatives_mask)) # This is FP + TN

    # --- Detection and Time to Detection ---
    true_incident_indices = np.where(y_true == 1)[0]
    
    if len(true_incident_indices) == 0:
        # Case with no true incident in the stream
        return {
            'detected': False, 
            'mttd_minutes': None,
            'false_alarms': false_alarms,
            'total_negatives': total_negatives
        }

    actual_incident_start_idx = true_incident_indices[0]
    # We need to find detections within the incident period.
    detection_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    is_detected = len(detection_indices) > 0
    
    if is_detected:
        first_detection_idx = detection_indices[0]
        time_to_detect_steps = first_detection_idx - actual_incident_start_idx
        mttd_minutes = float(time_to_detect_steps * resolution_minutes)
    else:
        mttd_minutes = None

    return {
        'detected': is_detected,
        'mttd_minutes': mttd_minutes,
        'false_alarms': false_alarms,
        'total_negatives': total_negatives
    }
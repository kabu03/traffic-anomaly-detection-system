import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

import numpy as np
import pandas as pd

from src.config import FEATURE_SETS, TRAIN_LEN, STREAM_LEN, HOLDOUT_LEN

def load_stream_data(file_path: str, feature_set_name: str, train_len: int = TRAIN_LEN):
    """
    Returns:
      X_train: first train_len rows (features only)
      X_full: full stream (features only)
      y_holdout: labels AFTER train_len (evaluation region)
      station_id: station ID of the stream (scalar)
    """
    df = pd.read_csv(file_path)
    features = FEATURE_SETS[feature_set_name]
    X_full = df[features].values
    X_train = X_full[:train_len]
    y_holdout = df['is_incident'].values[train_len:]
    station_id = df['station_id'].iloc[0]
    return X_train, X_full, y_holdout, station_id

def get_param_hash(params: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

def _incident_num_from_path(stream_path: str) -> int:
    stem = Path(stream_path).stem
    return int(stem.split('_')[-1])

def concat_X_train(stream_paths: List[str], feature_set: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate the TRAIN segments (first TRAIN_LEN rows) of multiple streams.

    Returns:
      X_train_concat: (sum(TRAIN_LEN), d)
      station_ids_concat: (sum(TRAIN_LEN),) per-row station_id
      incident_nums_concat: (sum(TRAIN_LEN),) per-row incident number (actual IDs like 9, 17, ...)
    """
    X_parts, S_parts, I_parts = [], [], []
    for p in stream_paths:
        df = pd.read_csv(p)
        feats = FEATURE_SETS[feature_set]
        X = df[feats].values[:TRAIN_LEN]
        s_ids = df['station_id'].values[:TRAIN_LEN]
        inc = _incident_num_from_path(p)
        X_parts.append(X.astype(np.float32))
        S_parts.append(s_ids)
        I_parts.append(np.full(X.shape[0], inc, dtype=int))
    X_train_concat = np.vstack(X_parts) if X_parts else np.empty((0, len(FEATURE_SETS[feature_set])), dtype=np.float32)
    station_ids_concat = np.concatenate(S_parts) if S_parts else np.empty((0,), dtype=object)
    incident_nums_concat = np.concatenate(I_parts) if I_parts else np.empty((0,), dtype=int)
    return X_train_concat, station_ids_concat, incident_nums_concat

def concat_X_full_and_y_holdout(stream_paths: List[str], feature_set: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Concatenate FULL streams for scoring, and collect holdout labels per stream.

    Returns:
      X_full_concat: (sum(STREAM_LEN), d)
      y_holdout_list: list of arrays, one per stream, each length HOLDOUT_LEN
      station_ids_full: (sum(STREAM_LEN),) per-row station_id
      incident_nums_full: (sum(STREAM_LEN),) per-row incident number
    """
    X_full_parts, s_full_parts, i_full_parts = [], [], []
    y_holdout_list: List[np.ndarray] = []
    for p in stream_paths:
        df = pd.read_csv(p)
        feats = FEATURE_SETS[feature_set]
        X_full = df[feats].values.astype(np.float32)
        y_hold = df['is_incident'].values[TRAIN_LEN:]
        s_full = df['station_id'].values
        inc = _incident_num_from_path(p)
        X_full_parts.append(X_full)
        s_full_parts.append(s_full)
        i_full_parts.append(np.full(X_full.shape[0], inc, dtype=int))
        y_holdout_list.append(y_hold.astype(int))
    X_full_concat = np.vstack(X_full_parts) if X_full_parts else np.empty((0, len(FEATURE_SETS[feature_set])), dtype=np.float32)
    station_ids_full = np.concatenate(s_full_parts) if s_full_parts else np.empty((0,), dtype=object)
    incident_nums_full = np.concatenate(i_full_parts) if i_full_parts else np.empty((0,), dtype=int)
    return X_full_concat, y_holdout_list, station_ids_full, incident_nums_full


def build_station_mapping(station_ids: Iterable[Any]) -> Dict[Any, int]:
    """
    Build mapping from station_id to index with UNK=0 reserved.
    Returns dict: {station_id: idx} with idx in [1..V], UNK implicitly 0.
    """
    unique_ids = pd.unique(pd.Series(list(station_ids)))
    # Assign 1..V to seen ids; 0 is reserved for UNK
    mapping = {sid: i + 1 for i, sid in enumerate(sorted(map(str, unique_ids)))}
    # If station_ids are numeric, str() above ensures consistent keys; consumers should str() too.
    return mapping


def build_station_feature(station_ids: Iterable[Any], mapping: Dict[Any, int], mode: str):
    """
    Build per-row station feature aligned to station_ids:
      - mode='ohe': returns one-hot matrix (N, V+1) where column 0 is UNK.
      - mode='embed': returns int indices column (N,1) with UNK=0.
    """
    s = np.array(list(station_ids))
    # Normalise key type to str to match mapping creation above
    idxs = np.array([mapping.get(str(v), 0) for v in s], dtype=np.int32)
    if mode == 'embed':
        return idxs.reshape(-1, 1)  # (N,1)
    if mode == 'ohe':
        vocab_size = (max(mapping.values()) if mapping else 0) + 1  # +1 to include UNK=0
        oh = np.zeros((len(idxs), vocab_size), dtype=np.float32)
        if len(idxs) > 0:
            oh[np.arange(len(idxs)), idxs] = 1.0
        return oh
    raise ValueError(f"Unknown mode '{mode}'. Expected 'ohe' or 'embed'.")


def compute_reporting_aggregate(per_stream_metrics: List[Dict[str, Any]]):
    total_correctly_detected = sum(1 for m in per_stream_metrics if m['detected'])
    total_incidents = sum(1 for m in per_stream_metrics if m['mttd_minutes'] is not None or m['detected'] is False)
    total_false_alarms = sum(m['false_alarms'] for m in per_stream_metrics)
    total_negatives = sum(m['total_negatives'] for m in per_stream_metrics)
    detected_mttds_minutes = [m['mttd_minutes'] for m in per_stream_metrics if m['detected']]

    final_dr = total_correctly_detected / total_incidents if total_incidents > 0 else 0.0
    final_far = total_false_alarms / total_negatives if total_negatives > 0 else 0.0
    final_mttd_minutes = float(np.mean(detected_mttds_minutes)) if detected_mttds_minutes else None

    return {'mean_dr': final_dr, 'mean_far': final_far, 'mean_mttd_minutes': final_mttd_minutes}


def aggregate_results(pipeline_type, model_name, feature_set_name, results_base_dir="results"):
    """
    Aggregates results for a given experiment IF AND ONLY IF all 5 iterations are complete
    and an entry does not already exist in the final monolithic summary.
    """
    print(f"--- Attempting to Aggregate: {pipeline_type}/{model_name}/{feature_set_name} ---")
    
    summary_file_path = Path(results_base_dir) / "final_summary.json"
    
    # 1. Load existing summary data or initialize if it doesn't exist
    if summary_file_path.exists():
        with open(summary_file_path, 'r') as f:
            summary_data = json.load(f)
    else:
        summary_data = {}

    # 2. Check if this experiment is already in the summary file
    if summary_data.get(pipeline_type, {}).get(model_name, {}).get(feature_set_name):
        print(f"✓ SKIPPING: Results for this experiment already exist in {summary_file_path}.")
        return

    # 3. Check if all 5 iteration result files exist
    print("Checking for completion of all 5 iterations.")
    result_files = []
    all_iterations_complete = True
    for iteration_id in range(1, 6):
        metrics_file = Path(f"{results_base_dir}/{pipeline_type}/iteration_{iteration_id}/{model_name}/{feature_set_name}/final_metrics.json")
        if metrics_file.exists():
            result_files.append(str(metrics_file))
        else:
            print(f"✗ FAILED: Results for Iteration {iteration_id} are missing.")
            all_iterations_complete = False
            break # No need to check other iterations

    if not all_iterations_complete:
        print("Cannot aggregate until all 5 iterations are complete. Aborting.")
        return

    print("SUCCESS: All 5 iterations are complete. Proceeding with aggregation.")

    # 4. Perform the aggregation (this code only runs if all checks pass)
    all_iteration_metrics = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            metrics = data.get('reporting_metrics_aggregated', {})
            metrics['total_tuning_time_seconds'] = data.get('tuning_results', {}).get('total_tuning_time_seconds')
            all_iteration_metrics.append(metrics)

    df = pd.DataFrame(all_iteration_metrics)
    
    # Calculate final mean values
    final_metrics = {
        "num_iterations_run": len(df),
        "mean_dr": df['mean_dr'].mean(),
        "mean_far": df['mean_far'].mean(),
        "mean_mttd_minutes": df['mean_mttd_minutes'].mean(),
        "mean_training_time_seconds": df['mean_training_time_seconds'].mean(),
        "mean_prediction_time_seconds": df['mean_prediction_time_seconds'].mean(),
        "mean_tuning_time_seconds": df['total_tuning_time_seconds'].mean()
    }

    # 5. Update the monolithic summary data structure
    summary_data.setdefault(pipeline_type, {}).setdefault(model_name, {})[feature_set_name] = final_metrics

    # 6. Write the updated data back to the monolithic file
    with open(summary_file_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ Successfully aggregated and updated {summary_file_path} with new results.")
    print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))
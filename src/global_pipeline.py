# Example usage:
# python3 -m src.global_pipeline --iteration 1 --model IsolationForest --features Speed

import json
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from tensorflow.keras import backend as K # type: ignore
import joblib

from src.config import MODEL_CONFIG, FEATURE_SETS, TRAIN_LEN
from src.thresholds import find_best_threshold
from src.metrics import compute_metrics
from src.data_utils import (
    concat_X_train,
    concat_X_full_and_y_holdout,
    build_station_mapping,
    build_station_feature,
    get_param_hash,
    compute_reporting_aggregate,
)

def _fit_model_for_split(model_class, params, is_dl, scaler, X_num_train, station_ids_train, incident_nums_train):
    """
    Fit a model for the single training split within an iteration.
    """
    station_map = build_station_mapping(station_ids_train)
    X_num_scaled = scaler.fit_transform(X_num_train).astype(np.float32)

    if is_dl:
        S_idx = build_station_feature(station_ids_train, station_map, mode='embed')
        model = model_class(**params)
        t0 = time.perf_counter()
        model.fit(X_num_scaled, station_idx=S_idx, incident_nums=incident_nums_train)
        fit_time = time.perf_counter() - t0
        return model, station_map, True, fit_time
    else:
        S_ohe = build_station_feature(station_ids_train, station_map, mode='ohe')
        X_train_final = np.hstack([X_num_scaled, S_ohe]).astype(np.float32)
        model = model_class(**params)
        t0 = time.perf_counter()
        model.fit(X_train_final)
        fit_time = time.perf_counter() - t0
        return model, station_map, False, fit_time

def _score_stream(model, is_dl, scaler, station_map, feature_set_name, stream_path):
    """
    Score one stream using the TRAIN-fitted scaler + station_map.
    Returns:
      scores_holdout (np.ndarray), y_holdout (np.ndarray), score_time_seconds (decision_function only)
    """
    X_full, y_holdout_list, station_ids_full, incident_nums_full = concat_X_full_and_y_holdout([stream_path], feature_set_name)
    y_holdout = y_holdout_list[0]
    X_full_scaled = scaler.transform(X_full).astype(np.float32)

    if is_dl:
        S_full_idx = build_station_feature(station_ids_full, station_map, mode='embed')
        t0 = time.perf_counter()
        scores_full = model.decision_function(X_full_scaled, station_idx=S_full_idx, incident_nums=incident_nums_full)
        score_time = time.perf_counter() - t0
    else:
        S_full_ohe = build_station_feature(station_ids_full, station_map, mode='ohe')
        X_full_final = np.hstack([X_full_scaled, S_full_ohe]).astype(np.float32)
        t0 = time.perf_counter()
        scores_full = model.decision_function(X_full_final)
        score_time = time.perf_counter() - t0

    scores_holdout = scores_full[TRAIN_LEN:]
    return scores_holdout, y_holdout, score_time

def _atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)

def process_global_param_set(params, model_class, iteration_data, feature_set_name, is_dl):
    """
    Evaluate a single hyperparameter set on the validation split of this iteration.
    """
    print(f"[GLOBAL TUNING] Params: {params}")

    all_val_scores, all_val_y_true = [], []
    compute_time = 0.0

    # Extract the single split from the iteration data
    train_streams = iteration_data['training_streams']
    val_streams = iteration_data['validation_streams']

    X_num_train, station_ids_train, incident_nums_train = concat_X_train(train_streams, feature_set_name)

    scaler = StandardScaler()
    model, station_map, used_embed, fit_time = _fit_model_for_split(
        model_class, params, is_dl, scaler, X_num_train, station_ids_train, incident_nums_train
    )
    compute_time += fit_time
    print(f"  Fit Time={fit_time:.2f}s | Val Streams={len(val_streams)}")

    for stream_path in val_streams:
        scores_holdout, y_holdout, score_time = _score_stream(
            model, used_embed, scaler, station_map, feature_set_name, stream_path
        )
        compute_time += score_time
        all_val_scores.append(scores_holdout)
        all_val_y_true.append(y_holdout)

    del model
    K.clear_session()
    gc.collect()

    tau, cost = find_best_threshold(all_val_scores, all_val_y_true)
    result = {'params': params, 'tau': tau, 'cost': cost}
    print(f"[GLOBAL TUNING COMPLETE] Params: {params} | Tau={tau:.4f} | Cost={cost:.2f} | ComputeTime={compute_time:.2f}s")
    return get_param_hash(params), result, compute_time

def _calibrate_tau_on_tuning(global_model, is_dl, scaler, station_map, feature_set_name, tuning_stream_paths):
    """
    Score the holdouts of the 140 tuning streams and optimize cost.
    Returns tau, cost, total_score_time_seconds (decision_function only).
    """
    all_scores, all_y = [], []
    total_score_time = 0.0
    for p in tuning_stream_paths:
        scores_holdout, y_holdout, score_time = _score_stream(global_model, is_dl, scaler, station_map, feature_set_name, p)
        all_scores.append(scores_holdout)
        all_y.append(y_holdout)
        total_score_time += score_time
    tau, cost = find_best_threshold(all_scores, all_y)
    return tau, cost, total_score_time

def _save_artifacts(artifacts_dir: Path, scaler: StandardScaler, station_map: dict, model, is_dl: bool):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Scaler
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    # Station map (keys are typically strings; ensure JSON serializable)
    _atomic_write_json(artifacts_dir / "station_map.json", station_map)
    # Model
    if is_dl and hasattr(model, "model_") and model.model_ is not None:
        # Keras model inside detector
        model.model_.save(artifacts_dir / "model.h5")
    else:
        # Best-effort for sklearn-like detectors
        try:
            joblib.dump(model, artifacts_dir / "model.pkl")
        except Exception as e:
            print(f"Warning: could not save model artifact ({e}).")

def run_global_pipeline(iteration_id, model_name, feature_set_name,
                        results_base_dir="results", overwrite=False,
                        n_jobs=1, param_hash=None):
    print("--- Running Global Pipeline ---")
    print("n_jobs = ", n_jobs)
    print(f"Iteration: {iteration_id}, Model: {model_name}, Features: {feature_set_name}")
    
    results_base_dir = Path(results_base_dir)
    # Updated path structure to match iteration
    result_dir = results_base_dir / f"global/iteration_{iteration_id}/{model_name}/{feature_set_name}"
    final_metrics_file = result_dir / "final_metrics.json"
    tuning_params_dir = result_dir / "tuning_params"   # per-param checkpoints
    artifacts_dir = result_dir / "artifacts"           # trained model + scaler + mapping

    if final_metrics_file.exists() and not overwrite:
        print("✓ Results already exist. Use --overwrite to run again. Skipping.")
        return
    elif final_metrics_file.exists() and overwrite:
        print("! Overwriting existing results.")

    result_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Stream Partitions ---
    with open("data/stream_partitions.json", 'r') as f:
        iteration_data = json.load(f)[f"iteration_{iteration_id}"]

    tuning_stream_paths = iteration_data['tuning_streams']       # 140 (Union of Train + Val)
    reporting_stream_paths = iteration_data['reporting_streams'] # 35
    # We pass the whole iteration_data to process_global_param_set to access training/validation_streams

    model_cfg = MODEL_CONFIG[model_name]
    model_class = model_cfg['class']
    is_dl = model_cfg.get('is_dl', False)

    full_grid = list(ParameterGrid(model_cfg['params']))

    if param_hash:
        matched = [p for p in full_grid if get_param_hash(p) == param_hash]
        if not matched:
            raise SystemExit(f"[ERROR] No hyperparameter set matches hash: {param_hash}")
        param_grid = matched
        print(f"[PARAM HASH MODE] Using only hash {param_hash}")
        print(f"[PARAM HASH MODE] Parameters: {matched[0]}")
    else:
        param_grid = full_grid

    print(f"Total hyperparameter combinations considered: {len(param_grid)}")

    tuning_summary_path = result_dir / "tuning_summary.json"

    # --- Load existing tuning cache ---
    if tuning_summary_path.exists() and not overwrite:
        with open(tuning_summary_path, 'r') as f:
            cached = json.load(f)
        all_tuning_results = cached.get('results', {})
        accumulated_time = cached.get('metadata', {}).get('total_tuning_time_seconds', 0.0)
        params_to_run = [p for p in param_grid if get_param_hash(p) not in all_tuning_results]
        print(f"Found {len(all_tuning_results)} cached results. Remaining: {len(params_to_run)}")
        print(f"Recovered prior tuning time: {accumulated_time:.2f}s")
    else:
        all_tuning_results = {}
        accumulated_time = 0.0
        params_to_run = param_grid

    # Merge per-param files
    per_param_files = []
    if tuning_params_dir.exists():
        for jf in tuning_params_dir.glob("*.json"):
            try:
                with open(jf, "r") as f:
                    obj = json.load(f)
                per_param_files.append(obj)
                ph = obj.get("hash")
                res = obj.get("result")
                if ph and res:
                    all_tuning_results.setdefault(ph, res)
            except Exception as e:
                print(f"Warning: could not read {jf.name}: {e}")
        params_to_run = [p for p in param_grid if get_param_hash(p) not in all_tuning_results]

        # If summary timing isn't available, reconstruct accumulated_time from per-param files
        if accumulated_time == 0.0 and per_param_files:
            accumulated_time = float(sum(obj.get("time_seconds", 0.0) for obj in per_param_files))
            print(f"Reconstructed tuning time from per-param files: {accumulated_time:.2f}s")

    # --- Step 1: Hyperparam + Threshold Tuning ---
    print(f"Step 1/4: Hyperparameter + Threshold Tuning (Single Split)...")
    if params_to_run:
        def _run_and_checkpoint(p):
            h, r, compute_t = process_global_param_set(p, model_class, iteration_data, feature_set_name, is_dl)
            _atomic_write_json(
                tuning_params_dir / f"{h}.json",
                {"hash": h, "result": r, "time_seconds": compute_t, "params": p}
            )
            return h, r, compute_t

        # SERIAL when n_jobs == 1, else parallel
        if n_jobs == 1:
            parallel_results = [_run_and_checkpoint(params) for params in params_to_run]
        else:
            parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, pre_dispatch='n_jobs')(
                delayed(_run_and_checkpoint)(params) for params in params_to_run
            )

        new_results = {h: r for (h, r, _) in parallel_results}
        new_time = sum(t for (_, _, t) in parallel_results)
        all_tuning_results.update(new_results)
        accumulated_time += new_time

        with open(tuning_summary_path, 'w') as f:
            json.dump({
                'metadata': {'total_tuning_time_seconds': accumulated_time},
                'results': all_tuning_results
            }, f, indent=2)
        print(f"[TUNE CACHE WRITE] Added {len(new_results)} new results.")
    else:
        print("No remaining hyperparameter sets to evaluate (all cached).")

    if not all_tuning_results:
        raise RuntimeError("No tuning results available. Aborting.")

    # Select best hyperparameters (min cost)
    best_hash = min(all_tuning_results, key=lambda h: all_tuning_results[h]['cost'])
    best_entry = all_tuning_results[best_hash]
    best_params = best_entry['params']
    validation_tau = best_entry['tau']
    best_cost = best_entry['cost']

    print("\n--- Hyperparameter Selection Complete ---")
    print(f"Best Params: {best_params}")
    print(f"Validation Tau: {validation_tau:.4f} | Validation Cost: {best_cost:.2f}")

    # --- Step 2: Train final global model on all 140 tuning streams ---
    print("Step 2/4: Training final global model on all tuning streams...")
    X_num_train_all, station_ids_train_all, incident_nums_train_all = concat_X_train(tuning_stream_paths, feature_set_name)
    scaler = StandardScaler()
    X_num_train_all_scaled = scaler.fit_transform(X_num_train_all).astype(np.float32)
    station_map_all = build_station_mapping(station_ids_train_all)

    if is_dl:
        S_idx_all = build_station_feature(station_ids_train_all, station_map_all, mode='embed')
        global_model = model_class(**best_params)
        global_train_start = time.perf_counter()
        global_model.fit(X_num_train_all_scaled, station_idx=S_idx_all, incident_nums=incident_nums_train_all)
        global_training_time = time.perf_counter() - global_train_start
    else:
        S_ohe_all = build_station_feature(station_ids_train_all, station_map_all, mode='ohe')
        X_train_full = np.hstack([X_num_train_all_scaled, S_ohe_all]).astype(np.float32)
        global_model = model_class(**best_params)
        global_train_start = time.perf_counter()
        global_model.fit(X_train_full)
        global_training_time = time.perf_counter() - global_train_start

    print(f"Global model training time: {global_training_time:.2f}s")
    # Save training artifacts so we can resume later without retraining
    _save_artifacts(artifacts_dir, scaler, station_map_all, global_model, is_dl)
    _atomic_write_json(result_dir / "best_selection.json", {
        "best_params": best_params,
        "validation_tau": validation_tau,
        "validation_cost": best_cost
    })

    # --- Step 3: Calibrate tau on the 140 tuning holdouts ---
    print("Step 3/4: Calibrating threshold (tau) on tuning holdouts...")
    calibrated_tau, calib_cost, calib_score_time = _calibrate_tau_on_tuning(
        global_model, is_dl, scaler, station_map_all, feature_set_name, tuning_stream_paths
    )
    print(f"Calibrated Tau: {calibrated_tau:.4f} | Calibration Cost: {calib_cost:.2f}")

    _atomic_write_json(result_dir / "calibration.json", {
        "calibrated_tau": calibrated_tau,
        "calibration_cost": calib_cost
    })

    # --- Step 4: Reporting on 35 streams ---
    print(f"Step 4/4: Evaluating global model on {len(reporting_stream_paths)} reporting streams...")
    reporting_metrics_list = []
    total_predict_time = 0.0
    # Divide by the number of streams we trained on, which is the length of the tuning set
    per_stream_train_time = global_training_time / len(tuning_stream_paths) if tuning_stream_paths else 0.0

    for stream_path in reporting_stream_paths:
        scores_holdout, y_holdout, score_time = _score_stream(
            global_model, is_dl, scaler, station_map_all, feature_set_name, stream_path
        )
        total_predict_time += score_time

        metrics = compute_metrics(y_holdout, scores_holdout, calibrated_tau)
        metrics['stream_path'] = stream_path
        metrics['training_time_seconds'] = per_stream_train_time
        metrics['prediction_time_seconds'] = score_time
        reporting_metrics_list.append(metrics)

    # --- Aggregate & Save ---
    print("Aggregating and saving results...")
    aggregate = compute_reporting_aggregate(reporting_metrics_list)
    avg_train_time = per_stream_train_time
    avg_predict_time = total_predict_time / len(reporting_stream_paths) if reporting_stream_paths else 0.0

    final_results = {
        'metadata': {
            'pipeline_type': 'global',
            'model_name': model_name,
            'feature_set': feature_set_name,
            'iteration_id': iteration_id,
            'run_timestamp_utc': datetime.utcnow().isoformat()
        },
        'tuning_results': {
            'best_hyperparameters': best_params,
            'validation_best_threshold': validation_tau,
            'best_threshold': calibrated_tau,
            'tuning_set_cost': best_cost,
            'calibration_cost': calib_cost,
            'total_tuning_time_seconds': accumulated_time
        },
        'reporting_metrics_aggregated': {
            **aggregate,
            'mean_training_time_seconds': avg_train_time,
            'mean_prediction_time_seconds': avg_predict_time
        },
        'reporting_metrics_per_stream': reporting_metrics_list,
        'timing_extras': {
            'global_training_time_seconds': global_training_time,
            'calibration_scoring_time_seconds': calib_score_time,
            'total_reporting_predict_time_seconds': total_predict_time
        }
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    final_metrics_file = result_dir / "final_metrics.json"
    with open(final_metrics_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"✓ Successfully saved results to {final_metrics_file}")
    print(pd.DataFrame([final_results['reporting_metrics_aggregated']]).round(4).to_string(index=False))

    # Cleanup
    del global_model
    K.clear_session()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the global model pipeline.")
    parser.add_argument('--iteration', type=int, required=True, help="Iteration ID (1-5).")
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CONFIG.keys(), help="Model to run.")
    parser.add_argument('--features', type=str, required=True, choices=FEATURE_SETS.keys(), help="Feature set to use.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing results.")
    parser.add_argument('--jobs', type=int, default=1, help="Number of parallel jobs for tuning (1 = serial).")
    parser.add_argument('--param-hash', type=str, help="Restrict tuning to a single hyperparameter set identified by this hash.")
    args = parser.parse_args()

    run_global_pipeline(
        iteration_id=args.iteration,
        model_name=args.model,
        feature_set_name=args.features,
        overwrite=args.overwrite,
        n_jobs=args.jobs,
        param_hash=args.param_hash
    )
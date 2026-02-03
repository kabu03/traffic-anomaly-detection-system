# Example usage: python3 -m src.individual_pipeline --iteration 1 --model LSTM --features Speed

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import time
import os
import gc
from joblib import Parallel, delayed
from tensorflow.keras import backend as K # type: ignore
from src.metrics import compute_metrics
from src.thresholds import find_best_threshold
from src.config import MODEL_CONFIG, FEATURE_SETS, TRAIN_LEN
from src.data_utils import load_stream_data, get_param_hash, compute_reporting_aggregate

def process_single_param_set(params, model_class, tuning_stream_paths, feature_set_name, score_cache_dir, overwrite):
    """
    Worker function to process a single hyperparameter combination.
    This function is designed to be run in parallel.
    """
    param_hash = get_param_hash(params)
    print(f"Processing params: {params}")

    all_scores_for_params = []
    all_y_true_for_params = []
    computation_time = 0.0

    for stream_path in tuning_stream_paths:
        stream_id = Path(stream_path).stem
        param_cache_dir = score_cache_dir / param_hash
        param_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = param_cache_dir / f"{stream_id}_scores.npy"

        if cache_path.exists() and not overwrite:
            # print(f"[CACHE HIT] {cache_path.name}")
            scores = np.load(cache_path)
            _, _, y_true, _ = load_stream_data(stream_path, feature_set_name)
        else:
            X_train, X_to_score, y_true, _ = load_stream_data(stream_path, feature_set_name)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_to_score_scaled = scaler.transform(X_to_score)
            
            model = model_class(**params)
            
            start_compute = time.perf_counter()
            model.fit(X_train_scaled)
            scores_full = model.decision_function(X_to_score_scaled)
            end_compute = time.perf_counter()
            
            computation_time += (end_compute - start_compute)
            
            # CRITICAL: Evaluate and cache scores for the holdout set ONLY
            scores = scores_full[TRAIN_LEN:]
            np.save(cache_path, scores)
            
            del model
            K.clear_session()
            gc.collect()

        all_scores_for_params.append(scores)
        all_y_true_for_params.append(y_true)

    tau_for_params, cost_for_params = find_best_threshold(
        all_scores_for_params, 
        all_y_true_for_params,
    )
    
    result = {
        'params': params,
        'tau': tau_for_params,
        'cost': cost_for_params
    }
    
    return param_hash, result, computation_time


def run_individual_pipeline(iteration_id, model_name, feature_set_name, results_base_dir="results", overwrite=False):
    print(f"--- Running Individual Pipeline ---")
    print(f"Iteration: {iteration_id}, Model: {model_name}, Features: {feature_set_name}")
    results_base_dir = Path(results_base_dir)

    # 1. SETUP: Define result path and check if already completed
    result_dir = results_base_dir / f"individual/iteration_{iteration_id}/{model_name}/{feature_set_name}"
    final_metrics_file = result_dir / "final_metrics.json"
    
    if final_metrics_file.exists() and not overwrite:
        print(f"✓ Results already exist. Use --overwrite to run again. Skipping.")
        return
    elif final_metrics_file.exists() and overwrite:
        print("! Overwriting existing results.")

    result_dir.mkdir(parents=True, exist_ok=True)

    # 2. LOAD DATA SPLITS
    with open("data/stream_partitions.json", 'r') as f:
        splits = json.load(f)[f"iteration_{iteration_id}"]
    
    tuning_stream_paths = splits['tuning_streams']
    reporting_stream_paths = splits['reporting_streams']

    # 3. HYPERPARAMETER TUNING
    print(f"Step 1/3: Tuning hyperparameters on {len(tuning_stream_paths)} streams...")
    model_class = MODEL_CONFIG[model_name]['class']
    param_grid = list(ParameterGrid(MODEL_CONFIG[model_name]['params']))
    
    score_cache_dir = results_base_dir / "cache" / model_name / feature_set_name
    score_cache_dir.mkdir(parents=True, exist_ok=True)
    
    tuning_summary_path = result_dir / "tuning_summary.json"
    print(f"Tuning summary will be saved to: {tuning_summary_path}")

    # --- Load existing tuning results and initialize computation time ---
    if tuning_summary_path.exists() and not overwrite:
        with open(tuning_summary_path, 'r') as f:
            cached_data = json.load(f)
        all_tuning_results = cached_data.get('results', {})
        # Load the previously saved time, defaulting to 0.0 if not found
        actual_tuning_computation_time = cached_data.get('metadata', {}).get('total_tuning_time_seconds', 0.0)
        params_to_run = [p for p in param_grid if get_param_hash(p) not in all_tuning_results]
        print(f"Found {len(all_tuning_results)} cached results. Running {len(params_to_run)} new parameter sets.")
        print(f"Loaded {actual_tuning_computation_time:.2f}s of prior computation time from cache.")
    else:
        all_tuning_results = {}
        params_to_run = param_grid
        actual_tuning_computation_time = 0.0 # Initialize for a fresh run

    # --- Parallel Execution using Joblib ---
    # Can tweak params like n_jobs
    if params_to_run:
        parallel_results = Parallel(n_jobs=6, backend="loky", verbose=10, pre_dispatch='n_jobs')(
            delayed(process_single_param_set)(
                params, model_class, tuning_stream_paths, feature_set_name, score_cache_dir, overwrite
            ) for params in params_to_run
        )

        # --- Collect results from the parallel runs ---
        new_results = {res[0]: res[1] for res in parallel_results}
        total_new_computation_time = sum(res[2] for res in parallel_results)
        
        all_tuning_results.update(new_results)
        
        # Add the newly computed time to the existing total
        actual_tuning_computation_time += total_new_computation_time

        # --- Save the updated summary file ---
        data_to_save = {
            'metadata': {'total_tuning_time_seconds': actual_tuning_computation_time},
            'results': all_tuning_results
        }
        with open(tuning_summary_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[TUNE CACHE WRITE] Updated tuning summary with {len(new_results)} new results.")

    # --- After the loop, find the best result from all the tuning runs ---
    if not all_tuning_results:
        raise RuntimeError("Tuning produced no results. Please check for errors.")

    # Find the hash corresponding to the minimum cost
    best_run_hash = min(all_tuning_results, key=lambda h: all_tuning_results[h]['cost'])
    
    best_result = all_tuning_results[best_run_hash]
    best_params = best_result['params']
    best_tau = best_result['tau']
    best_overall_cost = best_result['cost']

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best Parameters: {best_params}")
    print(f"Best Tau: {best_tau:.4f} (Avg Cost: {best_overall_cost:.2f})")

    # 4. REPORTING
    print(f"Step 2/3: Evaluating on {len(reporting_stream_paths)} reporting streams...")
    reporting_metrics_list = []
    total_reporting_train_time = 0
    total_reporting_predict_time = 0
    for stream_path in reporting_stream_paths:
        X_train, X_to_score, y_test_holdout, _ = load_stream_data(stream_path, feature_set_name)
        
        # --- ADD SCALING STEP ---
        scaler = StandardScaler()

        # Fit the scaler ONLY on the training data and transform it
        X_train_scaled = scaler.fit_transform(X_train)

        # Use the SAME scaler (already fitted) to transform the data to be scored
        X_to_score_scaled = scaler.transform(X_to_score)
        # --- END SCALING STEP ---
        
        # Train a new model with the best hyperparams on this stream's training data
        model = model_class(**best_params)

        start_time = time.perf_counter()
        model.fit(X_train_scaled)
        train_time = time.perf_counter() - start_time
        total_reporting_train_time += train_time

        start_time = time.perf_counter()
        scores_full = model.decision_function(X_to_score_scaled)
        predict_time = time.perf_counter() - start_time
        total_reporting_predict_time += predict_time
        
        # Evaluate using the best threshold on the holdout set ONLY
        scores_holdout = scores_full[TRAIN_LEN:]
        metrics = compute_metrics(y_test_holdout, scores_holdout, best_tau)

        # Add timing info to the per-stream metrics
        metrics['training_time_seconds'] = train_time
        metrics['prediction_time_seconds'] = predict_time
        reporting_metrics_list.append(metrics)

    # 5. AGGREGATE AND SAVE RESULTS
    print("Step 3/3: Aggregating and saving results...")

    aggregate = compute_reporting_aggregate(reporting_metrics_list)

    avg_train_time = total_reporting_train_time / len(reporting_stream_paths) if reporting_stream_paths else 0
    avg_predict_time = total_reporting_predict_time / len(reporting_stream_paths) if reporting_stream_paths else 0
    
    final_results = {
        'metadata': {
            'pipeline_type': 'individual',
            'model_name': model_name,
            'feature_set': feature_set_name,
            'iteration_id': iteration_id,
            'run_timestamp_utc': datetime.utcnow().isoformat()
        },
        'tuning_results': {
            'best_hyperparameters': best_params,
            'best_threshold': best_tau,
            'tuning_set_cost': best_overall_cost,
            'total_tuning_time_seconds': actual_tuning_computation_time
        },
        'reporting_metrics_aggregated': {
            **aggregate,
            'mean_training_time_seconds': avg_train_time,
            'mean_prediction_time_seconds': avg_predict_time
        },
        'reporting_metrics_per_stream': reporting_metrics_list
    }

    with open(final_metrics_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str) # Use default=str for Path objects etc.
        
    print(f"✓ Successfully saved results to {final_metrics_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the individual model pipeline.")
    parser.add_argument('--iteration', type=int, required=True, help="Iteration ID (1-5).")
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CONFIG.keys(), help="Model to run.")
    parser.add_argument('--features', type=str, required=True, choices=FEATURE_SETS.keys(), help="Feature set to use.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing results.")
    
    args = parser.parse_args()
    
    run_individual_pipeline(
        iteration_id=args.iteration,
        model_name=args.model,
        feature_set_name=args.features,
        overwrite=args.overwrite
    )
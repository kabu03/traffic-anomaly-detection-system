import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

# Constants
RESULTS_DIR = Path("results")
STREAM_PARTITIONS_PATH = Path("data/stream_partitions.json")

def load_stream_partitions() -> Dict[str, Any]:
    """Loads the experiment configuration (streams per iteration)."""
    if not STREAM_PARTITIONS_PATH.exists():
        # Throw error
        raise FileNotFoundError(f"Stream partitions were not found: {STREAM_PARTITIONS_PATH}")
        
    with open(STREAM_PARTITIONS_PATH, 'r') as f:
        return json.load(f)

def load_global_summary() -> Dict[str, Any]:
    """Loads the monolithic final_summary.json."""
    path = RESULTS_DIR / "final_summary.json"
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def get_available_runs() -> List[Dict[str, str]]:
    """
    Scans the results directory to find which experiments have been run.
    Returns a list of dicts: [{'pipeline': 'global', 'model': 'LSTM', 'feature': 'Speed'}, ...]
    """
    runs = []
    for pipeline in ['global', 'individual']:
        pipe_dir = RESULTS_DIR / pipeline
        if not pipe_dir.exists():
            continue
            
        # We look at iteration_1 to determine available models/features
        # Assuming if it ran for iter 1, it's a valid experiment configuration
        iter1 = pipe_dir / "iteration_1"
        if iter1.exists():
            for model_dir in iter1.iterdir():
                if model_dir.is_dir():
                    for feat_dir in model_dir.iterdir():
                        if feat_dir.is_dir() and (feat_dir / "final_metrics.json").exists():
                            runs.append({
                                'pipeline': pipeline,
                                'model': model_dir.name,
                                'feature': feat_dir.name
                            })
    return runs

def load_run_metrics(pipeline: str, model: str, feature: str, iteration: int) -> Optional[Dict[str, Any]]:
    """
    Loads final_metrics.json and normalizes differences between pipelines.
    Specifically, it injects 'stream_path' into individual pipeline results.
    """
    path = RESULTS_DIR / pipeline / f"iteration_{iteration}" / model / feature / "final_metrics.json"
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # --- Normalization Logic ---
    
    # 1. Reverse Engineer Stream Paths for Individual Pipeline
    if pipeline == 'individual':
        stream_partitions = load_stream_partitions()
        iter_key = f"iteration_{iteration}"
        
        if iter_key in stream_partitions:
            reporting_streams = stream_partitions[iter_key].get('reporting_streams', [])
            metrics_list = data.get('reporting_metrics_per_stream', [])
            
            # The individual pipeline processes streams in the exact order of the stream_partitions list
            if len(reporting_streams) == len(metrics_list):
                for i, metric in enumerate(metrics_list):
                    metric['stream_path'] = reporting_streams[i]
            else:
                # Fallback if counts mismatch (shouldn't happen if data is clean)
                for i, metric in enumerate(metrics_list):
                    metric['stream_path'] = f"unknown_stream_index_{i}"

    # 2. Normalize Tuning Cost Terminology
    # Global uses 'validation_cost', Individual uses 'tuning_set_cost'
    tuning_res = data.get('tuning_results', {})
    if 'validation_cost' in tuning_res:
        tuning_res['unified_cost'] = tuning_res['validation_cost']
    elif 'tuning_set_cost' in tuning_res:
        tuning_res['unified_cost'] = tuning_res['tuning_set_cost']
        
    return data

def load_tuning_curve(pipeline: str, model: str, feature: str, iteration: int) -> Optional[pd.DataFrame]:
    """
    Loads tuning_summary.json.
    Returns None for Individual pipeline (as it doesn't exist there).
    """
    path = RESULTS_DIR / pipeline / f"iteration_{iteration}" / model / feature / "tuning_summary.json"
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        rows = []
        for param_hash, res in results.items():
            row = res['params'].copy()
            row['cost'] = res['cost']
            row['tau'] = res['tau']
            row['hash'] = param_hash
            rows.append(row)
        
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error loading tuning curve for {pipeline}, {model}, {feature}, iter {iteration}: {e}")
        return None
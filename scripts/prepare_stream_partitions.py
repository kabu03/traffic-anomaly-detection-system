import json
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
from src.config import SPECIFIC_INCIDENT_NUMS

def prepare_stream_partitions(
    stream_ids: list,
    n_folds: int = 5,
    outer_seed: int = 54321,
    inner_seed: int = 12345,
    output_file: Path = Path('./data/stream_partitions.json'),
    data_dir: str = 'data/incident_files/smoothed'
):
    """
    Generates reproducible experiment iterations.
    Structure:
        Iteration N:
            - Reporting Streams (35/175)
            - Tuning Streams (140/175)
                -> Training Streams (112/140, i.e., 80%), only used in the Global Pipeline.
                -> Validation Streams (28/140, i.e., 20%), only used in the Global Pipeline.
    """
    print("Generating stream partitions...")
    
    stream_ids = np.array(stream_ids)
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
    
    stream_dictionary = {}

    for i, (tuning_indices, reporting_indices) in enumerate(outer_cv.split(stream_ids)):
        iteration_key = f"iteration_{i+1}" # 1-based indexing
        
        tuning_stream_ids = stream_ids[tuning_indices]
        reporting_stream_ids = stream_ids[reporting_indices]
        
        rng = np.random.RandomState(inner_seed + i)
        perm = rng.permutation(len(tuning_stream_ids))
        n_val = max(1, int(round(0.2 * len(tuning_stream_ids))))
        
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        train_stream_ids = tuning_stream_ids[train_indices]
        val_stream_ids = tuning_stream_ids[val_indices]

        # Flat Structure
        stream_dictionary[iteration_key] = {
            'reporting_streams': [f"{data_dir}/incident_{id}.csv" for id in reporting_stream_ids],
            'tuning_streams':    [f"{data_dir}/incident_{id}.csv" for id in tuning_stream_ids],
            'training_streams':  [f"{data_dir}/incident_{id}.csv" for id in train_stream_ids],
            'validation_streams':[f"{data_dir}/incident_{id}.csv" for id in val_stream_ids]
        }
            
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stream_dictionary, f, indent=2)
        
    print(f"\nSuccessfully generated and saved stream partitions to {output_file}")

if __name__ == '__main__':
    prepare_stream_partitions(stream_ids=SPECIFIC_INCIDENT_NUMS)
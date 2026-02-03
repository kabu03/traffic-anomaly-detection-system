import json
import shutil
import os
from pathlib import Path

RESULTS_DIR = Path("results")

KEY_MAPPING = {
    "inner_cv_cost": "validation_cost",
    "inner_cv_tau": "validation_tau",
    "clip_id": "stream_id",
    "clip_path": "stream_path",
    "num_folds_found": "num_iterations_run",
    "avg_predict_time_seconds": "mean_prediction_time_seconds",
    "avg_train_time_seconds": "mean_training_time_seconds",
    "outer_fold_id": "iteration_id",
    "inner_cv_best_threshold": "validation_best_threshold",
    # Other mappings should be added here if needed
}

ORDERED_KEYS = [
    "num_iterations_run",
    "mean_dr",
    "mean_far",
    "mean_mttd_minutes",
    "mean_training_time_seconds",
    "mean_prediction_time_seconds",
    "mean_tuning_time_seconds"
]

def migrate_json_content(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON: {file_path}")
        return

    def recursive_update(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                k_lower = k.lower()
                
                # --- Logic 1: outer_fold_id -> iteration_id (Value + 1) ---
                if k == "outer_fold_id":
                    new_k = "iteration_id"
                    # Increment 0-based index to 1-based iteration
                    new_v = (v + 1) if isinstance(v, int) else v
                    new_obj[new_k] = new_v
                    continue

                # --- Logic 2: MTTD Hours -> Minutes (* 60) ---
                # Matches "mean_mttd_hours", "MTTD_hours", "mttd_hours" etc.
                if "mttd" in k_lower and "hours" in k_lower:
                    # Replace 'hours' with 'minutes' preserving case
                    new_k = k.replace("hours", "minutes").replace("Hours", "Minutes")
                    
                    # Convert value
                    if isinstance(v, (int, float)):
                        new_v = v * 60
                    else:
                        new_v = v
                    
                    new_obj[new_k] = recursive_update(new_v)
                    continue

                # --- Logic 3: Standard Renaming ---
                new_k = KEY_MAPPING.get(k, k)
                new_k = new_k.replace("clip", "stream") # Generic clip->stream
                
                new_obj[new_k] = recursive_update(v)
            
            # --- Logic 4: Reordering ---
            # Only reorder if it looks like a metrics dictionary
            if any(key in new_obj for key in ["mean_dr", "dr", "DR"]):
                ordered_obj = {}
                # Add priority keys
                for key in ORDERED_KEYS:
                    if key in new_obj:
                        ordered_obj[key] = new_obj.pop(key)
                # Add remaining keys
                ordered_obj.update(new_obj)
                return ordered_obj
            
            return new_obj
            
        elif isinstance(obj, list):
            return [recursive_update(i) for i in obj]
        else:
            return obj

    new_data = recursive_update(data)

    with open(file_path, 'w') as f:
        json.dump(new_data, f, indent=2)

def migrate_directory_structure():
    # Rename Folders: fold_X -> iteration_X+1
    # We walk top-down but sort reverse to handle nested folders safely if needed
    fold_dirs = sorted(list(RESULTS_DIR.rglob("fold_*")), key=lambda p: str(p), reverse=True)
    
    for d in fold_dirs:
        if not d.is_dir(): continue
        
        parts = d.name.split('_')
        # Check if it matches "fold_N" exactly
        if len(parts) == 2 and parts[0] == 'fold' and parts[1].isdigit():
            fold_idx = int(parts[1])
            new_name = f"iteration_{fold_idx + 1}"
            new_path = d.parent / new_name
            
            if not new_path.exists():
                print(f"Renaming Directory: {d.name} -> {new_name}")
                d.rename(new_path)
            else:
                print(f"Skipping rename {d.name} -> {new_name} (Target exists)")

def migrate_file_contents():
    for json_file in RESULTS_DIR.rglob("*.json"):
        # print(f"Processing file: {json_file.name}")
        migrate_json_content(json_file)

def main():
    if not RESULTS_DIR.exists():
        print("Results directory not found.")
        return
    
    print("Migrating Directory Structure...")
    migrate_directory_structure()

    print("Migrating File Contents...")
    migrate_file_contents()

    print("\nMigration Complete.")

if __name__ == "__main__":
    main()
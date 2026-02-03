import subprocess
import sys
from pathlib import Path

# --- Direct Function Imports ---
try:
    from src.config import MODEL_CONFIG, FEATURE_SETS
    from src.data_utils import aggregate_results
except ImportError as e:
    print(f"Error: Could not import a required module: {e}")
    print("Please ensure this script is run from the project's root directory and all required files exist.")
    sys.exit(1)

def get_user_choice(prompt, options, allow_all=False):
    """
    Generic function to display a menu and get a validated user choice.
    Returns a list of selected options.
    """
    print(f"\n{prompt}")
    display_options = list(options) # Make a copy

    if allow_all:
        display_options.append("All of the above")

    for i, option in enumerate(display_options, 1):
        print(f"  {i}) {option}")
    
    while True:
        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(options): # Meaning the user chose a specific option
                return [options[choice - 1]]
            elif allow_all and choice == len(display_options): # Meaning the user chose "All of the above"
                return list(options)
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """Main function to drive the interactive experiment orchestration."""
    print("--- Experiment Orchestrator ---")

    model_list = list(MODEL_CONFIG.keys())
    chosen_model = get_user_choice("Select a model to run:", model_list)[0]

    feature_list = list(FEATURE_SETS.keys())
    chosen_features_list = get_user_choice("Select a feature set:", feature_list, allow_all=True)

    # Let the user choose whether they want the individual or global model pipeline.
    chosen_pipeline = get_user_choice("Select a pipeline type:", ["individual", "global"])[0]

    overwrite_choice = input("\nOverwrite existing final results and re-compute all cached scores? (y/n): ").strip().lower()
    should_overwrite = (overwrite_choice == 'y')
    if should_overwrite:
        print("! Overwrite mode enabled. All cached scores will be re-computed.")
        print("Are you sure you want to proceed? (y/n): ")
        confirm = input().strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    else:
        print("✓ Overwrite mode disabled. Existing cache will be used.")

    print("\n--- Summary of Jobs to Run ---")
    print(f"  Model:      {chosen_model}")
    print(f"  Features:   {chosen_features_list}")
    print(f"  Pipeline:   {chosen_pipeline}")
    print("   Iterations:  All (1-5, managed by robust runner)")
    print("--------------------------------")

    confirm = input("Proceed with running these jobs? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return

    # --- Main Execution Loop ---
    for chosen_features in chosen_features_list:
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {chosen_pipeline.upper()} / {chosen_model} / {chosen_features}")
        print(f"{'='*60}")

        # Build the command to execute the robust shell script
        command = ['./run_robust.sh', chosen_model, chosen_features, chosen_pipeline]
        if should_overwrite:
            command.append('--overwrite')

        try:
            # Execute the shell script and wait for it to complete
            # This will run all 5 iterations robustly.
            process = subprocess.run(command, check=True, text=True)
            
            # --- Automatic Aggregation ---
            # This runs only after the robust runner has successfully completed all iterations.
            print("\n>>> Automatically aggregating results for the completed experiment... <<<")
            aggregate_results(
                pipeline_type=chosen_pipeline,
                model_name=chosen_model,
                feature_set_name=chosen_features
            )

        except FileNotFoundError:
            print("\n!!! ERROR: 'run_robust.sh' not found. Make sure it's in the root directory and executable.")
            break
        except subprocess.CalledProcessError as e:
            # This catches errors from within the shell script (e.g., python script crashing)
            print(f"\n!!! ERROR: The robust runner script failed for {chosen_features}. See output above.")
            print("Aborting remaining jobs.")
            break
        except Exception as e:
            print(f"\n!!! An unexpected error occurred: {e}")
            break

    print(f"\n{'='*60}")
    print("--- Orchestrator Finished All Jobs ---")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
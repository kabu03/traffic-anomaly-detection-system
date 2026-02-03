import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import argparse
from pathlib import Path
from src.config import SPECIFIC_INCIDENT_NUMS 
# --- Configuration & Constants ---
DATA_DIR = Path('ai-powered-traffic-ad-system/data')
INCIDENT_DATA_DIR = DATA_DIR / 'incident_files'
PLOT_OUTPUT_BASE_DIR = DATA_DIR / 'incident_plots'

def visualize_timeseries(df, feature, incident_id, sensor_id, save_path, is_smoothed):
    """Generates a time-series plot from a pre-processed dataframe."""
    plt.figure(figsize=(18, 6))
    
    feature_label = feature.capitalize()
    plot_color = 'darkorange' if feature == 'occ' else 'blue'
    
    if is_smoothed:
        smoothed_col = f"{feature}_smoothed"
        plt.plot(df['timestamp'], df[smoothed_col], marker='', linestyle='-', color=plot_color, label=f'Smoothed {feature_label}')
        title = f'Incident {incident_id} at Sensor {sensor_id}: Smoothed Upstream {feature_label}'
    else:
        plt.plot(df['timestamp'], df[feature], marker='.', linestyle='-', markersize=4, color=plot_color, label=f'Raw {feature_label}')
        title = f'Incident {incident_id} at Sensor {sensor_id}: Raw Upstream {feature_label}'

    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(feature_label)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    incident_period = df[df['is_incident'] == 1]
    if not incident_period.empty:
        plt.axvspan(incident_period['timestamp'].min(), incident_period['timestamp'].max(), color='gray', alpha=0.5, label='Incident Period')

    if not df.empty:
        test_period_start_time = df['timestamp'].iloc[0] + pd.Timedelta(days=10)
        plt.axvline(x=test_period_start_time, color='r', linestyle='--', linewidth=2, label='Test Period Start')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Main Execution Logic ---

def main(mode, ids_to_process=None):
    """Main function to drive the plot generation based on the selected mode."""
    is_smoothed = 'smoothed' in mode
    data_subdir = 'smoothed' if is_smoothed else 'raw'
    input_data_dir = INCIDENT_DATA_DIR / data_subdir

    if ids_to_process:
        incident_nums = ids_to_process
        print(f"Processing specific incident IDs from command line: {ids_to_process}")
    elif 'all' in mode:
        incident_nums = sorted([int(f.stem.split('_')[1]) for f in input_data_dir.glob('incident_*.csv')])
    else: # Meaning it's 'specific'
        incident_nums = SPECIFIC_INCIDENT_NUMS

    output_dir = PLOT_OUTPUT_BASE_DIR / mode
    if ids_to_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
    print(f"Reading data from: {input_data_dir}")
    print(f"Output will be saved to: {output_dir}")

    for i in incident_nums:
        file_path = input_data_dir / f'incident_{i}.csv'
        if not file_path.exists():
            print(f"Skipping Incident {i}: File not found in {data_subdir} directory.")
            continue
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            if df.empty:
                print(f"Skipping Incident {i}: File is empty.")
                continue
            
            df = df.sort_values('timestamp')
            sensor_id = df['station_id'].iloc[0]
            print(f"Processing Incident {i} (Sensor {sensor_id})...")

            for feature in ['speed', 'occ']:
                plot_filename = f"incident_{i}_{feature}.png"
                save_path = output_dir / plot_filename
                visualize_timeseries(df, feature, i, sensor_id, save_path, is_smoothed=is_smoothed)

        except Exception as e:
            print(f"ERROR processing Incident {i}: {e}")

    print(f"\nPlot generation complete for mode '{mode}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from pre-processed incident data files.")
    parser.add_argument(
        'mode',
        choices=['raw_all', 'raw_specific', 'smoothed_all', 'smoothed_specific'],
        help="The operating mode, determining which data to read and where to save plots."
    )
    parser.add_argument(
        '--ids',
        type=int,
        nargs='+',
        help="A specific list of incident IDs to process, overriding the mode's default list."
    )
    args = parser.parse_args()
    main(args.mode, ids_to_process=args.ids)
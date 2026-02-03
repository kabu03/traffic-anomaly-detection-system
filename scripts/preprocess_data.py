import pandas as pd
import os
from datetime import timedelta
import yaml
import re
import argparse
from pathlib import Path

def parse_timedelta(time_str):
    """Parses a time string like '-1h', '+30m' into a timedelta object."""
    if not time_str or not isinstance(time_str, str):
        return timedelta()
    
    parts = re.match(r"([+-]?)(\d+)([hms])", time_str)
    if not parts:
        return timedelta()
        
    sign, value, unit = parts.groups()
    value = int(value)

    if sign == '-':
        value = -value
    if unit == 'h':
        return timedelta(hours=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    
    return timedelta()

def generate_incident_files(ids_to_process=None, alpha=0.15):
    """
    Loads data, extracts traffic data for each incident, and saves both a raw
    and a smoothed version of the data to separate directories.
    """

    data_dir = Path('../data')
    incidents_file = data_dir / 'incidents.csv'
    traffic_file = data_dir / 'traffic_data.parquet'
    base_output_dir = data_dir / 'incident_files'
    raw_output_dir = base_output_dir / 'raw'
    smoothed_output_dir = base_output_dir / 'smoothed'
    trims_file = 'trims.yaml'

    # Create output directories
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    smoothed_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raw files will be saved to: {raw_output_dir}")
    print(f"Smoothed files will be saved to: {smoothed_output_dir}")

    print("Loading datasets...")
    incidents_df = pd.read_csv(incidents_file)
    traffic_df = pd.read_parquet(traffic_file)
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    print("Datasets loaded.")

    trims = {}
    if os.path.exists(trims_file):
        with open(trims_file, 'r') as f:
            trims = yaml.safe_load(f) or {}
        print(f"Loaded {len(trims)} manual trims from {trims_file}")

    TRAINING_DAYS = 10
    TOTAL_TEST_HOURS = 12

    if ids_to_process:
        incidents_to_process_df = incidents_df.loc[incidents_df.index.isin(ids_to_process)]
        print(f"Processing specific incident IDs: {ids_to_process}")
    else:
        incidents_to_process_df = incidents_df
        print("Processing all incidents.")

    for index, incident in incidents_to_process_df.iterrows():
        incident_ts = incident['timestamp']
        up_sensor_id = incident['up_id']
        
        pre_incident_hours = 6
        trim_offset = parse_timedelta(trims.get(index))
        if trim_offset != timedelta():
            print(f"  -> Incident {index}: Applying manual trim of {trims.get(index)}")

        test_period_start_ts = incident_ts - timedelta(hours=pre_incident_hours)
        start_time = test_period_start_ts - timedelta(days=TRAINING_DAYS) + trim_offset
        end_time = test_period_start_ts + timedelta(hours=TOTAL_TEST_HOURS) + trim_offset

        incident_traffic_data = traffic_df[
            (traffic_df['station_id'] == up_sensor_id) &
            (traffic_df['timestamp'] >= start_time) &
            (traffic_df['timestamp'] <= end_time)
        ].copy()

        if incident_traffic_data.empty:
            print(f"No traffic data found for incident {index}. File not created.")
            continue

        incident_traffic_data['is_incident'] = 0
        incident_end_ts = incident_ts + timedelta(minutes=incident['duration'])
        incident_period_mask = (incident_traffic_data['timestamp'] >= incident_ts) & \
                               (incident_traffic_data['timestamp'] < incident_end_ts)
        incident_traffic_data.loc[incident_period_mask, 'is_incident'] = 1

        output_filename = f"incident_{index}.csv"

        # --- Save Raw File ---
        incident_traffic_data.to_csv(raw_output_dir / output_filename, index=False)

        # --- Create and Save Smoothed File ---
        smoothed_df = incident_traffic_data.copy()
        smoothed_df['speed_smoothed'] = smoothed_df['speed'].ewm(alpha=alpha, adjust=False).mean()
        smoothed_df['occ_smoothed'] = smoothed_df['occ'].ewm(alpha=alpha, adjust=False).mean()
        smoothed_df.to_csv(smoothed_output_dir / output_filename, index=False)
        
        print(f"Successfully created raw and smoothed files for incident {index}")

    print("\nProcessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate raw and smoothed incident CSV files.")
    parser.add_argument(
        '--ids', type=int, nargs='+',
        help="A space-separated list of specific incident IDs to process."
    )
    parser.add_argument(
        '--alpha', type=float, default=0.15,
        help="The smoothing factor (alpha) for the exponential moving average."
    )
    args = parser.parse_args()
    generate_incident_files(ids_to_process=args.ids, alpha=args.alpha)
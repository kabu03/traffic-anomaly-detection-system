import sys
from pathlib import Path
import streamlit as st

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.results_loader import (
    load_global_summary, 
    get_available_runs, 
    load_run_metrics, 
    load_tuning_curve
)

ACRONYMS = {
    "ModZScore": "MZ",
    "IsolationForest": "IF",
    "LOF": "LOF",
    "OneClassSVM": "OC-SVM",
    "LSTM": "LSTM"
}

MODEL_ORDER = ["MZ", "IF", "LOF", "OC-SVM", "LSTM"]

FEATURE_COLORS = {
    "Speed": "#1f77b4",      # blue
    "Occupancy": "#ff7f0e",  # orange
    "Bivariate": "#2ca02c",  # green
}

def get_acronym(model_name):
    return ACRONYMS.get(model_name, model_name)

def load_and_process_summary():
    """Loads summary and returns a processed DataFrame."""
    summary_data = load_global_summary()
    if not summary_data:
        return None

    rows = []
    for pipeline, models in summary_data.items():
        for model, features in models.items():
            if "Autoencoder" in model:
                continue

            for feature_set, metrics in features.items():
                dr = metrics.get("mean_dr", 0)
                far = metrics.get("mean_far", 0)
                mttd = metrics.get("mean_mttd_minutes", 0)
                cost = 100 * (1 - dr) + 100 * far + mttd

                row = {
                    "Pipeline": pipeline.capitalize(),
                    "Model Name": model,
                    "Model": get_acronym(model),
                    "Feature Set": feature_set,
                    "mean_dr": dr,
                    "mean_dr_pct": dr * 100,
                    "mean_far": far,
                    "mean_far_pct": far * 100,
                    "mean_mttd_minutes": mttd,
                    "mean_training_time_seconds": metrics.get("mean_training_time_seconds"),
                    "mean_prediction_time_seconds": metrics.get("mean_prediction_time_seconds"),
                    "cost": cost
                }
                rows.append(row)
    return rows

def apply_custom_style():
    st.markdown("""
        <style>
        /* 1. Increase Global Font Size */
        html, body, [class*="css"]  {
            font-size: 18px; 
        }
        
        /* 2. Style the Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f1f3f6;
            border-right: 1px solid #dce0e6;
        }
        
        /* 3. KPI Card Styles (Shared) */
        .metric-card {
            background-color: #011f4b;
            border: 0px solid #262730;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%;
        }
        .metric-label { 
            font-size: 18px; 
            color: #e0e0e0; 
            margin-bottom: 5px; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value { 
            font-size: 28px; 
            font-weight: bold; 
            color: white; 
            margin: 5px 0;
        }
        .metric-sub { 
            font-size: 14px; 
            color: #b3cde0; 
            font-weight: 500; 
            margin-top: 5px; 
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)

def display_kpi_card(col, title, main_value, sub_info):
    """
    Renders a styled KPI card in the given column.
    """
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{main_value}</div>
            <div class="metric-sub">{sub_info}</div>
        </div>
        """, unsafe_allow_html=True)
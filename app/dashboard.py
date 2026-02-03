import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.utils import apply_custom_style

from app.views.intro import render_intro
from app.views.performance import render_performance_overview
from app.views.analysis import render_analysis
from app.views.evolution import render_evolution
from app.views.comparator import render_comparator

st.set_page_config(page_title="AI-Powered Traffic Anomaly Detection System", layout="wide")

# Inject CSS early
apply_custom_style()

st.markdown("""
    <style>
        .centered-title { text-align: center; margin-bottom: 0px; }
        .subtitle-block { 
            text-align: center; 
            margin-top: 10px;   /* reduces spacing below title */
            margin-bottom: 30px; /* small spacing before menu */
        }
    </style>

    <h1 class='centered-title'>AI-Powered Traffic Anomaly Detection System</h1>
""", unsafe_allow_html=True)

# --- Navigation ---
selected = option_menu(
    menu_title=None,
    options=[
        "Introduction",
        "Performance",
        "Analysis",
        "Evolution",
        "Comparator",
    ],
    icons=["info-circle", "trophy", "bar-chart-line", "search", "sliders"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#005b96"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-family": "monospace", "font-size": "16px", "text-align": "center", "margin":"0px"},
        "nav-link-selected": {"background-color": "#011f4b"},
    }
)

# --- Routing ---
if selected == "Introduction":
    render_intro()
elif selected == "Performance":
    render_performance_overview()
elif selected == "Analysis":
    render_analysis()
elif selected == "Evolution":
    render_evolution()
elif selected == "Comparator":
    render_comparator()
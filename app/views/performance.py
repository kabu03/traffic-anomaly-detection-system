import streamlit as st
import pandas as pd
from app.utils import load_and_process_summary, apply_custom_style, FEATURE_COLORS, MODEL_ORDER, display_kpi_card

def render_performance_overview():
    st.title("Performance Overview")

    rows = load_and_process_summary()
    if not rows:
        st.error("No data found.")
        st.stop()

    df = pd.DataFrame(rows)
    df['Model_Sort_Key'] = df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    df = df.sort_values(['Model_Sort_Key', 'Pipeline', 'Feature Set'])

    st.space()

    best_dr = df.loc[df['mean_dr'].idxmax()]
    best_far = df.loc[df['mean_far'].idxmin()]
    best_mttd = df.loc[df['mean_mttd_minutes'].idxmin()]

    k1, k2, k3 = st.columns(3)

    def format_sub(row):
        return f"{row['Pipeline']} / {row['Model']}<br>{row['Feature Set']}"

    display_kpi_card(k1, "Best Detection Rate", f"{best_dr['mean_dr_pct']:.2f}%", format_sub(best_dr))
    display_kpi_card(k2, "Best False Alarm Rate", f"{best_far['mean_far_pct']:.2f}%", format_sub(best_far))
    display_kpi_card(k3, "Best MTTD", f"{best_mttd['mean_mttd_minutes']:.2f} min", format_sub(best_mttd))

    st.divider()

    # --- Main Table ---
    st.subheader("Detailed Results Table")
    st.markdown("The table is interactive: You can scroll to explore, drag columns to resize them, and click any column header to sort the results.")
    def highlight_pipeline(val):
        if val == "Global":
            return 'background-color: #0984e3; color: white'
        elif val == "Individual":
            return 'background-color: #d63031; color: white'
        return ''

    def highlight_features(val):
        color = FEATURE_COLORS.get(val, "")
        if color:
            return f'color: {color}; font-weight: bold'
        return ''

    display_df = df[[
        "Pipeline", "Model", "Feature Set", 
        "mean_dr_pct", "mean_far_pct", "mean_mttd_minutes", 
        "mean_training_time_seconds", "mean_prediction_time_seconds"
    ]]

    st.dataframe(
        display_df.style
            .map(highlight_pipeline, subset=["Pipeline"])
            .map(highlight_features, subset=["Feature Set"])
            .format({
                "mean_dr_pct": "{:.2f}%",
                "mean_far_pct": "{:.2f}%",
                "mean_mttd_minutes": "{:.2f}",
                "mean_training_time_seconds": "{:.4f}",
                "mean_prediction_time_seconds": "{:.4f}"
            }),
        column_config={
            "mean_dr_pct": "DR (%)",
            "mean_far_pct": "FAR (%)",
            "mean_mttd_minutes": "MTTD (min)",
            "mean_training_time_seconds": "Train Time (s)",
            "mean_prediction_time_seconds": "Pred Time (s)"
        },
        width='stretch',
        hide_index=True,
        height=800
    )
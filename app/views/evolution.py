import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils import get_available_runs, load_run_metrics, get_acronym, apply_custom_style, display_kpi_card

def render_evolution():
    st.title("Evolution Analysis")
    st.markdown("Analyze how model performance and hyperparameters evolve across different iterations in the experiment.")

    available_runs = get_available_runs()
    available_runs = [r for r in available_runs if "Autoencoder" not in r['model']]

    if not available_runs:
        st.error("No valid experiment runs found.")
        st.stop()

    # --- Configuration Selectors ---
    c1, c2, c3 = st.columns(3)
    with c1:
        raw_pipelines = sorted(list(set(r['pipeline'] for r in available_runs)))
        pipeline_map = {p.capitalize(): p for p in raw_pipelines}
        selected_pipeline_display = st.selectbox("Pipeline", list(pipeline_map.keys()))
        selected_pipeline = pipeline_map[selected_pipeline_display]

    with c2:
        models_for_pipe = sorted(list(set(r['model'] for r in available_runs if r['pipeline'] == selected_pipeline)))

        # Default to "LSTM" if available
        default_model_ix = models_for_pipe.index("LSTM") if "LSTM" in models_for_pipe else 0

        selected_model = st.selectbox("Model", models_for_pipe, index=default_model_ix)
    with c3:
        feats_for_model = sorted(list(set(r['feature'] for r in available_runs if r['pipeline'] == selected_pipeline and r['model'] == selected_model)))
        
        # Default to "Speed" if available
        default_feat_ix = feats_for_model.index("Speed") if "Speed" in feats_for_model else 0
        
        selected_feature = st.selectbox("Feature Set", feats_for_model, index=default_feat_ix)

    st.divider()

    # --- Data Loading ---
    iter_metrics = []
    for i in range(1, 6):
        data = load_run_metrics(selected_pipeline, selected_model, selected_feature, i)
        if data:
            m = data['reporting_metrics_aggregated']
            tuning = data.get('tuning_results', {})
            best_params = tuning.get('best_hyperparameters', {})
            
            # Handle empty hyperparameters gracefully
            if not best_params:
                param_display = "This method does not inherently have any tunable hyperparameters."
            else:
                param_display = str(best_params)

            row = {
                "Iteration": i,
                "DR (%)": m.get("DR", m.get("mean_dr", 0)) * 100,
                "FAR (%)": m.get("FAR", m.get("mean_far", 0)) * 100,
                "MTTD (min)": m.get("MTTD_minutes", m.get("mean_mttd_minutes")),
                "Best Hyperparameters": param_display
            }
            iter_metrics.append(row)

    if not iter_metrics:
        st.warning("No data found for this configuration.")
        st.stop()

    df_iters = pd.DataFrame(iter_metrics)

    # --- Section 1: Stability Statistics ---
    st.subheader(f"Hyperparameter & Metric Evolution ({selected_pipeline.capitalize()} / {get_acronym(selected_model)})")
    
    m1, m2, m3 = st.columns(3)
    display_kpi_card(m1, "Average Detection Rate", f"{df_iters['DR (%)'].mean():.2f}%", f"σ = {df_iters['DR (%)'].std():.2f}")
    display_kpi_card(m2, "Average False Alarm Rate", f"{df_iters['FAR (%)'].mean():.2f}%", f"σ = {df_iters['FAR (%)'].std():.2f}")
    display_kpi_card(m3, "Average MTTD", f"{df_iters['MTTD (min)'].mean():.2f} min", f"σ = {df_iters['MTTD (min)'].std():.2f}")
    
    # --- Charts ---
    
    # Chart 1: DR & FAR
    fig_perf = go.Figure()

    # Add DR Line
    fig_perf.add_trace(go.Scatter(
        x=df_iters["Iteration"], y=df_iters["DR (%)"],
        mode='lines+markers', name='Detection Rate (%)',
        line=dict(color='#1f77b4', width=3), marker=dict(size=10)
    ))

    # Add FAR Line
    fig_perf.add_trace(go.Scatter(
        x=df_iters["Iteration"], y=df_iters["FAR (%)"],
        mode='lines+markers', name='False Alarm Rate (%)',
        line=dict(color='#ff7f0e', width=3), marker=dict(size=10)
    ))

    fig_perf.update_layout(
        title="DR & FAR Stability",
        xaxis_title="Iteration",
        yaxis_title="Percentage (%)",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(dtick=1) # Force integer ticks on X-axis
    )

    # Chart 2: MTTD
    fig_mttd = go.Figure()
    
    fig_mttd.add_trace(go.Scatter(
        x=df_iters["Iteration"], y=df_iters["MTTD (min)"],
        mode='lines+markers', name='MTTD (min)',
        line=dict(color='#2ca02c', width=3), marker=dict(size=10)
    ))

    fig_mttd.update_layout(
        title="MTTD Stability",
        xaxis_title="Iteration",
        yaxis_title="Minutes",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(dtick=1)
    )
    
    # Layout: Two charts side-by-side
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.plotly_chart(fig_perf, width='stretch')
        
    with col_chart2:
        st.plotly_chart(fig_mttd, width='stretch')
        
    # Table below
    st.markdown("##### Configuration Details")
    st.dataframe(
        df_iters[["Iteration", "Best Hyperparameters"]],
        hide_index=True,
        width='stretch',
    )
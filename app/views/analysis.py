import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils import load_and_process_summary, FEATURE_COLORS, MODEL_ORDER, apply_custom_style

def render_analysis():
    st.title("Comparative Analysis")

    rows = load_and_process_summary()
    if not rows:
        st.error("No data found.")
        st.stop()

    df = pd.DataFrame(rows)
    df['Model_Sort_Key'] = df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    df = df.sort_values(['Model_Sort_Key', 'Pipeline', 'Feature Set'])

    # 1. Scatter Plot (With Filtering)
    st.subheader("Detection Rate vs. False Alarm Rate")

    # Filter out Global LOF due to extreme FAR, and Global IF due to extremely low DR
    scatter_df = df[~((df['Pipeline'] == 'Global') & (df['Model'] == 'LOF')) & ~((df['Pipeline'] == 'Global') & (df['Model'] == 'IF'))]
    fig_scatter = px.scatter(
        scatter_df, x="mean_far_pct", y="mean_dr_pct", 
        color="Feature Set", symbol="Pipeline", text="Model", color_discrete_map=FEATURE_COLORS,
        title="Mean DR (%) vs Mean FAR (%) (Top-Left is Better)", height=800,
        # Rename hover labels
        labels={
            "mean_far_pct": "FAR (%)",
            "mean_dr_pct": "DR (%)"
        }
    )
    fig_scatter.update_traces(
        textposition='top center',
        textfont=dict(size=14),
        marker=dict(size=16)
    )

    fig_scatter.update_layout(
        xaxis_title="Mean FAR (%)", 
        xaxis_range=[0, 42],
        yaxis_title="Mean DR (%)", 
        yaxis_range=[89, 101],
        font=dict(size=16),
        legend=dict(
            font=dict(size=16),
            title=dict(font=dict(size=18)),
            yanchor="top",
            y=0.37,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("""
            ⚠️ **Note:** The Global LOF and Global IF models have been excluded from 
            this scatter plot as the FAR of the Global LOF model (~60%) 
            and DR of the Global IF model (~40%) make them practically undeployable and distort the scale.
            """)

    def create_faceted_bar(data, y_col, title, y_label):
        fig = px.bar(
            data, x="Model", y=y_col, color="Feature Set",
            facet_row="Pipeline", barmode="group",
            # Reorder Pipeline (Individual first/top)
            category_orders={
                "Model": MODEL_ORDER,
                "Pipeline": ["Individual", "Global"] 
            },
            color_discrete_map=FEATURE_COLORS, 
            title=title, 
            height=700,
            # Rename hover labels
            labels={
                y_col: y_label
            }
        )
        
        fig.update_layout(
            yaxis_title=y_label, 
            xaxis_title="Model", 
            font=dict(size=18),
            legend=dict(font=dict(size=18), title=dict(font=dict(size=20)))
        )
        
        fig.update_yaxes(title_text=y_label, matches=None) 
        
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig

    # 2. Bar Charts
    st.subheader("Metric Comparisons")
    t1, t2, t3 = st.tabs(["Detection Rate", "False Alarm Rate", "MTTD"])

    with t1:
        st.plotly_chart(create_faceted_bar(df, "mean_dr_pct", "Detection Rate (Higher is Better)", "DR (%)"), width='stretch')
    with t2:
        st.plotly_chart(create_faceted_bar(df, "mean_far_pct", "False Alarm Rate (Lower is Better)", "FAR (%)"), width='stretch')
    with t3:
        st.plotly_chart(create_faceted_bar(df, "mean_mttd_minutes", "Time to Detection (Lower is Better)", "MTTD (min)"), width='stretch')
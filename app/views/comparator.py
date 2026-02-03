import streamlit as st
import pandas as pd
from app.utils import load_and_process_summary, apply_custom_style

def render_comparator():

    st.title("⚖️ Comparator")
    st.markdown("""
    Select two different configurations below to compare their performance metrics side-by-side.  
    The center column highlights the difference and indicates which configuration performs better for each metric.
    """)
    st.space()
    rows = load_and_process_summary()
    if not rows:
        st.error("No data found.")
        st.stop()

    df = pd.DataFrame(rows)

    def get_config_selection(key_prefix):
        """Helper to create a set of dropdowns for selecting a specific config."""
        c1, c2, c3 = st.columns(3)
        with c1:
            pipe = st.selectbox("Pipeline", sorted(df['Pipeline'].unique()), key=f"{key_prefix}_pipe")
        with c2:
            models = df[df['Pipeline'] == pipe]['Model Name'].unique()
            model = st.selectbox("Model", models, key=f"{key_prefix}_model")
        with c3:
            feats = df[(df['Pipeline'] == pipe) & (df['Model Name'] == model)]['Feature Set'].unique()
            feat = st.selectbox("Feature Set", feats, key=f"{key_prefix}_feat")
        
        row = df[
            (df['Pipeline'] == pipe) & 
            (df['Model Name'] == model) & 
            (df['Feature Set'] == feat)
        ]
        
        if row.empty:
            return None
        return row.iloc[0]

    # --- Layout ---
    col_left, col_mid, col_right = st.columns([1, 0.1, 1])

    with col_left:
        st.subheader("Configuration A")
        row_a = get_config_selection("A")

    with col_right:
        st.subheader("Configuration B")
        row_b = get_config_selection("B")

    st.divider()

    if row_a is None or row_b is None:
        st.warning("Invalid configuration selected.")
        st.stop()

    # --- Comparison Metrics ---
    st.header("Direct Comparison")

    # CSS for the comparison table
    st.markdown("""
    <style>
    .comp-row {
        display: flex;
        align-items: center;
        padding: 15px 0;
        border-bottom: 1px solid #eee;
    }
    /* Flex: 1 ensures all 4 columns (Label, A, Mid, B) take equal width */
    .comp-col-a { flex: 1; text-align: center; font-size: 22px; font-weight: bold; color: #1f77b4; }
    .comp-col-b { flex: 1; text-align: center; font-size: 22px; font-weight: bold; color: #ff7f0e; }
    .comp-col-mid { flex: 1; text-align: center; font-size: 16px; color: #555; }
    .comp-label { flex: 1; font-size: 20px; font-weight: 600; padding-left: 20px;}
    
    .better-arrow { font-weight: bold; font-size: 18px; }
    .green-text { color: #2ca02c; }
    </style>
    """, unsafe_allow_html=True)

    def render_comparison_row(label, val_a, val_b, suffix="", inverse=False):
        diff = val_a - val_b
        abs_diff = abs(diff)
        
        # Determine winner
        if inverse: # Lower is better (FAR, MTTD)
            if val_a < val_b:
                winner = "A"
                arrow = "← A Wins"
            elif val_b < val_a:
                winner = "B"
                arrow = "B Wins →"
            else:
                winner = "Tie"
                arrow = "="
        else: # Higher is better (DR)
            if val_a > val_b:
                winner = "A"
                arrow = "← A Wins"
            elif val_b > val_a:
                winner = "B"
                arrow = "B Wins →"
            else:
                winner = "Tie"
                arrow = "="

        display_diff = f"{abs_diff:.2f}{suffix}"
        
        st.markdown(f"""
        <div class="comp-row">
            <div class="comp-label">{label}</div>
            <div class="comp-col-a">{val_a:.2f}{suffix}</div>
            <div class="comp-col-mid">
                <span class="green-text better-arrow">{arrow}</span><br>
                <span style="font-size: 14px; color: #888;">Diff: {display_diff}</span>
            </div>
            <div class="comp-col-b">{val_b:.2f}{suffix}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Header Row ---
    st.markdown("""
    <div class="comp-row" style="border-bottom: 2px solid #ddd; margin-bottom: 10px;">
        <div class="comp-label"></div> <!-- Empty spacer for the Label column -->
        <div class="comp-col-a" style="font-size: 18px; color: #1f77b4;">Config A</div>
        <div class="comp-col-mid green-text" style="font-size: 18px;font-weight: bold;">Comparison</div>
        <div class="comp-col-b" style="font-size: 18px; color: #ff7f0e;">Config B</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Rows
    render_comparison_row("Detection Rate", row_a['mean_dr_pct'], row_b['mean_dr_pct'], "%", inverse=False)
    render_comparison_row("False Alarm Rate", row_a['mean_far_pct'], row_b['mean_far_pct'], "%", inverse=True)
    render_comparison_row("MTTD", row_a['mean_mttd_minutes'], row_b['mean_mttd_minutes'], " min", inverse=True)
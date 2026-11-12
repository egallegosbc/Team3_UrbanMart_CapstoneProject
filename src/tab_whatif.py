"""Tab 4 - Real-Time What-If Scenario Inference Engine."""

import pandas as pd
import streamlit as st

from src import app_helpers as helpers

MONETARY_STEP = 100_000
HISTORY_LIMIT = 15
# Widget keys stored in st.session_state (one place, easy to check).
NUMERIC_KEYS = {"wi_recency": "Recency", "wi_frequency": "Frequency", "wi_monetary": "Monetary"}
CATEGORY_KEYS = {"wi_region": "Region", "wi_age": "CustomerAgeGroup",
                 "wi_gender": "CustomerGender", "wi_category": "PreferredCategory"}


@st.cache_data
def typical_profiles():
    """Median RFM and most frequent categories per K-Means segment (from the data)."""
    table = helpers.load_customer_table()
    artifacts = helpers.load_artifacts()
    rfm = table[["Recency", "Frequency_log", "Monetary_log"]]
    clusters = artifacts["kmeans"].predict(artifacts["kmeans_scaler"].transform(rfm))
    table = table.assign(Segment=[artifacts["segment_map"][c] for c in clusters])
    profiles = {}
    for segment, group in table.groupby("Segment"):
        profile = {key: int(group[col].median()) for key, col in NUMERIC_KEYS.items()}
        profile["wi_monetary"] = round(profile["wi_monetary"] / MONETARY_STEP) * MONETARY_STEP
        profile.update({key: group[col].mode().iloc[0] for key, col in CATEGORY_KEYS.items()})
        profiles[segment] = profile
    return profiles


def init_state(table):
    """Set default widget values once, BEFORE the widgets are created."""
    defaults = {key: int(table[col].median()) for key, col in NUMERIC_KEYS.items()}
    defaults["wi_monetary"] = round(defaults["wi_monetary"] / MONETARY_STEP) * MONETARY_STEP
    defaults.update({key: table[col].mode().iloc[0] for key, col in CATEGORY_KEYS.items()})
    defaults.update({"wi_model": "Logistic Regression", "wi_threshold": 0.5, "wi_history": []})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(segment):
    """Button callback: runs before the rerun, so widgets read the new values safely."""
    st.session_state.update(typical_profiles()[segment])


def clear_history():
    """Button callback that empties the scenario history."""
    st.session_state["wi_history"] = []


def render_inputs(table):
    """Draw the sliders and selectboxes. Ranges come from the customer table."""
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Recency (days since last purchase)", int(table["Recency"].min()),
                  int(table["Recency"].max()), key="wi_recency")
        st.slider("Frequency (number of purchases)", int(table["Frequency"].min()),
                  int(table["Frequency"].max()), key="wi_frequency")
        low = int(table["Monetary"].min()) // MONETARY_STEP * MONETARY_STEP
        high = (int(table["Monetary"].max()) // MONETARY_STEP + 1) * MONETARY_STEP
        st.slider("Monetary (total spend)", low, high, step=MONETARY_STEP, key="wi_monetary")
    with col2:
        for key, col in CATEGORY_KEYS.items():
            st.selectbox(col, sorted(table[col].unique()), key=key)
        st.selectbox("Model", list(helpers.CLASSIFIER_FILES), key="wi_model")
        st.slider("Decision threshold", 0.05, 0.95, step=0.05, key="wi_threshold")


def show_alert(level, message):
    """Show the policy message with the matching Streamlit alert type."""
    {"error": st.error, "warning": st.warning, "success": st.success}[level](message)


def log_scenario(row):
    """Append the scenario to the history only when it changed."""
    history = st.session_state["wi_history"]
    if not history or history[-1] != row:
        history.append(row)
        del history[:-HISTORY_LIMIT]


def render_whatif_tab():
    """Interactive scenario simulator using the serialized models."""
    table = helpers.load_customer_table()
    artifacts = helpers.load_artifacts()
    init_state(table)
    state = st.session_state

    st.markdown("Move the sliders to simulate a customer. Every change reloads the saved "
                "preprocessor and model and recomputes the churn probability live.")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Preset: typical Champion", on_click=apply_preset, args=("Champions",))
    b2.button("Preset: typical At Risk", on_click=apply_preset, args=("At Risk",))

    render_inputs(table)

    row = helpers.make_input_row(state["wi_recency"], state["wi_frequency"],
                                 state["wi_monetary"], state["wi_region"], state["wi_age"],
                                 state["wi_gender"], state["wi_category"])
    probability = helpers.predict_churn(artifacts, state["wi_model"], row)
    segment = helpers.predict_segment(artifacts, row)
    high_spend = table["Monetary"].quantile(helpers.HIGH_SPEND_QUANTILE)
    level, message = helpers.decide_policy(probability, state["wi_threshold"], segment,
                                           state["wi_monetary"], high_spend)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{probability:.1%}",
              delta=f"{probability - state['wi_threshold']:+.1%} vs threshold",
              delta_color="inverse")
    m2.metric("K-Means segment", segment)
    m3.metric("Threshold", f"{state['wi_threshold']:.2f}", help=f"Model: {state['wi_model']}")
    st.progress(min(max(probability, 0.0), 1.0), text=f"P(churn) = {probability:.3f}")
    show_alert(level, message)
    st.caption(f"Rules: P >= threshold -> re-engagement voucher (stronger if spend >= "
               f"{high_spend:,.0f}, the 75th percentile); threshold - {helpers.MIDDLE_BAND} "
               "<= P < threshold -> watch list; otherwise low risk (VIP voucher if Champions).")

    log_scenario({"Recency": state["wi_recency"], "Frequency": state["wi_frequency"],
                  "Monetary": state["wi_monetary"], "Region": state["wi_region"],
                  "Age": state["wi_age"], "Gender": state["wi_gender"],
                  "Category": state["wi_category"], "Model": state["wi_model"],
                  "P(churn)": round(probability, 3), "Segment": segment, "Alert": level})
    st.subheader("Scenario history")
    st.dataframe(pd.DataFrame(state["wi_history"]), width="stretch", hide_index=True)
    st.button("Clear history", on_click=clear_history)

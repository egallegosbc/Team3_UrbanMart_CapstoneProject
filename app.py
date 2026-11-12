"""
UrbanMart Customer Intelligence - Streamlit dashboard (Milestone 3).

Run from the repository root:   python -m streamlit run app.py

Layout: KPI cards on top, then 4 tabs. Each tab lives in its own module in src/.
All models are loaded from models/*.joblib at runtime (see src/app_helpers.py).
"""

import sys
from pathlib import Path

import streamlit as st

# Make "src" importable no matter which folder the app is launched from.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src import app_helpers as helpers  # noqa: E402
from src.tab_benchmark import render_benchmark_tab  # noqa: E402
from src.tab_eda import render_eda_tab  # noqa: E402
from src.tab_preprocessing import render_preprocessing_tab  # noqa: E402
from src.tab_whatif import render_whatif_tab  # noqa: E402

st.set_page_config(page_title="UrbanMart Customer Intelligence", page_icon=":bar_chart:",
                   layout="wide")

# Small CSS touch so the KPI cards look like cards in the dark theme.
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1A1F2B; border: 1px solid #2A3142;
    border-radius: 10px; padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


def render_kpis(transactions, metadata, benchmark):
    """Show the six KPI metric cards above the tabs."""
    classifiers = benchmark.dropna(subset=["ROC-AUC"])
    best = classifiers.loc[classifiers["ROC-AUC"].idxmax()]
    sizes = metadata["kmeans"]["segment_sizes"]
    at_risk_share = sizes["At Risk"] / sum(sizes.values())

    cols = st.columns(6)
    cols[0].metric("Customers", f"{transactions['CustomerID'].nunique():,}")
    cols[1].metric("Transactions", f"{len(transactions):,}")
    cols[2].metric("Total revenue", f"{transactions['TransactionValue'].sum() / 1e9:,.1f} B")
    cols[3].metric("Churn rate (target)", f"{metadata['class_balance']['churn_rate']:.1%}",
                   help=f"No purchase in the {metadata['window']['window_days']} days after "
                        f"{metadata['window']['cutoff_date']}.")
    cols[4].metric("At Risk share (K-Means)", f"{at_risk_share:.1%}")
    cols[5].metric("Best model (ROC-AUC)", f"{best['ROC-AUC']:.3f}", help=best["Model"])
    st.caption(f"Best model by test ROC-AUC: **{best['Model']}**. Revenue in local currency.")


def main():
    """Build the page: title, artifact check, KPIs and the four tabs."""
    st.title("UrbanMart Customer Intelligence")
    st.caption("Team 3 - Retention and CRM Manager dashboard - target: churn_risk_flag")

    # Stop early with a clear message if the serialized pipeline is missing.
    problems = helpers.check_artifacts()
    if problems:
        st.error("The serialized models are not ready.\n\n- " + "\n- ".join(problems)
                 + f"\n\n{helpers.RETRAIN_HINT}")
        st.stop()

    transactions = helpers.load_transactions()
    metadata = helpers.load_metadata()
    benchmark = helpers.load_report("benchmark_metrics.csv")
    render_kpis(transactions, metadata, benchmark)

    tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer and EDA", "Preprocessing and Correlation",
                                      "Multi-Model Benchmark", "What-If Inference"])
    with tab1:
        render_eda_tab()
    with tab2:
        render_preprocessing_tab()
    with tab3:
        render_benchmark_tab()
    with tab4:
        render_whatif_tab()


main()

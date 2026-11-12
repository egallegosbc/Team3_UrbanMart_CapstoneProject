"""Tab 1 - Data Explorer and EDA."""

import plotly.express as px
import streamlit as st

from src import app_helpers as helpers

HISTOGRAM_FEATURES = ["Recency", "Frequency", "Monetary", "Frequency_log", "Monetary_log"]


def churn_rate_bar(table, column):
    """Bar chart of the churn rate (%) for each value of a categorical column."""
    rates = (table.groupby(column)["churn_risk_flag"].mean() * 100).reset_index()
    fig = px.bar(rates, x=column, y="churn_risk_flag", template=helpers.PLOT_TEMPLATE,
                 labels={"churn_risk_flag": "Churn rate (%)"},
                 color_discrete_sequence=[helpers.ACCENT], title=f"Churn rate by {column}")
    overall = table["churn_risk_flag"].mean() * 100
    fig.add_hline(y=overall, line_dash="dash", annotation_text=f"overall {overall:.1f}%")
    return fig


def render_eda_tab():
    """Raw data sample, summary statistics and interactive EDA charts."""
    transactions = helpers.load_transactions()
    table = helpers.load_customer_table()

    st.subheader("Raw transactions (sample)")
    st.caption(f"{len(transactions):,} rows, {transactions.shape[1]} columns, from "
               f"{transactions['TransactionDate'].min().date()} to "
               f"{transactions['TransactionDate'].max().date()}.")
    st.dataframe(transactions.head(200), width="stretch", height=250)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction summary statistics")
        st.dataframe(transactions.describe(include="all").T, width="stretch")
    with col2:
        st.subheader("Customer table summary statistics")
        st.caption("One row per customer, features computed before the cutoff date.")
        st.dataframe(table.describe().T, width="stretch")

    st.subheader("Customer feature distribution by churn_risk_flag")
    feature = st.selectbox("Feature", HISTOGRAM_FEATURES, key="eda_feature")
    plot_df = table.assign(churn_risk_flag=table["churn_risk_flag"].astype(str))
    fig = px.histogram(plot_df, x=feature, color="churn_risk_flag", nbins=40,
                       barmode="overlay", histnorm="percent", opacity=0.65,
                       template=helpers.PLOT_TEMPLATE, color_discrete_map=helpers.CHURN_COLORS,
                       labels={"churn_risk_flag": "churn (1 = yes)"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Histograms are normalized to percent within each class, so the small churn "
               "class (13%) can be compared with the large non-churn class.")

    col3, col4 = st.columns(2)
    col3.plotly_chart(churn_rate_bar(table, "Region"), width="stretch")
    col4.plotly_chart(churn_rate_bar(table, "PreferredCategory"), width="stretch")

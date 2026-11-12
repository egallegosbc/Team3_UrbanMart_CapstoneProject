"""Tab 2 - Preprocessing and Correlation Matrix."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_helpers as helpers


def scaled_numeric(table, artifacts, metadata):
    """Apply the SAVED numeric pipeline (imputer + StandardScaler) to the table."""
    numeric_cols = metadata["numeric_features"]
    numeric_pipe = artifacts["preprocessor"].named_transformers_["num"]
    return pd.DataFrame(numeric_pipe.transform(table[numeric_cols]), columns=numeric_cols)


def render_preprocessing_tab():
    """Show data quality, log transform, scaling, one-hot encoding and correlation."""
    transactions = helpers.load_transactions()
    table = helpers.load_customer_table()
    metadata = helpers.load_metadata()
    artifacts = helpers.load_artifacts()
    window = metadata["window"]

    st.info(f"**Temporal split (no leakage).** Features use only purchases before "
            f"{window['cutoff_date']}. The target churn_risk_flag = 1 means the customer made "
            f"no purchase in the next {window['window_days']} days (until {window['last_date']}). "
            "The model never sees the future window, so Recency can be a feature without "
            "revealing the answer.")

    # 1) Missing values and duplicates
    st.subheader("1. Missing values and duplicates")
    col1, col2 = st.columns([2, 1])
    missing = transactions.isna().sum().rename("Missing values").to_frame()
    col1.dataframe(missing.T, width="stretch")
    quality = metadata["data_quality"]
    col2.metric("Duplicate rows", quality["duplicate_rows"])
    col2.metric("Duplicate TransactionIDs", quality["duplicate_transaction_ids"])
    st.caption("No missing values or duplicates were found. The pipeline still includes a "
               "median imputer (numeric) and a most-frequent imputer (categorical) as the "
               "documented treatment for future data.")

    # 2) Log transform
    st.subheader("2. Log transform (log1p) of Frequency and Monetary")
    raw_col = st.radio("Variable", ["Monetary", "Frequency"], horizontal=True, key="prep_log")
    col3, col4 = st.columns(2)
    col3.plotly_chart(px.histogram(table, x=raw_col, nbins=40, template=helpers.PLOT_TEMPLATE,
                                   title=f"{raw_col} (raw)"), width="stretch")
    col4.plotly_chart(px.histogram(table, x=f"{raw_col}_log", nbins=40,
                                   template=helpers.PLOT_TEMPLATE,
                                   title=f"{raw_col}_log = log1p({raw_col})",
                                   color_discrete_sequence=[helpers.ACCENT]),
                      width="stretch")
    skew = table[[raw_col, f"{raw_col}_log"]].skew().round(3)
    st.caption(f"Skewness before: {skew.iloc[0]}, after: {skew.iloc[1]}. The log compresses "
               "very large values (spend is in the tens of millions).")

    # 3) StandardScaler
    st.subheader("3. StandardScaler effect (saved preprocessor, fitted on train only)")
    scaled = scaled_numeric(table, artifacts, metadata)
    col5, col6 = st.columns(2)
    col5.markdown("**Before scaling**")
    col5.dataframe(table[metadata["numeric_features"]].describe().round(3),
                   width="stretch")
    col6.markdown("**After scaling** (mean close to 0, std close to 1)")
    col6.dataframe(scaled.describe().round(3), width="stretch")

    # 4) One-hot encoding
    st.subheader("4. One-Hot encoding of categorical features")
    names = metadata["transformed_feature_names"]
    st.caption(f"{len(metadata['categorical_features'])} categorical columns become "
               f"{len(names) - len(metadata['numeric_features'])} binary columns. "
               f"Final model input: {len(names)} columns.")
    matrix = artifacts["preprocessor"].transform(table.head(8))
    st.dataframe(pd.DataFrame(matrix, columns=names).round(3), width="stretch")

    # 5) Correlation heatmap
    st.subheader("5. Pearson correlation matrix (including the target)")
    include_dummies = st.checkbox("Include one-hot columns", value=False, key="prep_dummies")
    corr_df = table[["Recency", "Frequency", "Monetary", "Frequency_log", "Monetary_log",
                     "churn_risk_flag"]]
    if include_dummies:
        dummies = pd.get_dummies(table[metadata["categorical_features"]], dtype=int)
        corr_df = pd.concat([corr_df, dummies], axis=1)
    corr = corr_df.corr(method="pearson")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    template=helpers.PLOT_TEMPLATE, aspect="auto")
    fig.update_layout(height=650 if include_dummies else 500)
    st.plotly_chart(fig, width="stretch")
    target_corr = corr["churn_risk_flag"].drop("churn_risk_flag").abs().max()
    st.caption(f"Largest absolute correlation with churn_risk_flag: {target_corr:.3f}. "
               "Values this close to 0 mean no single feature explains churn linearly.")

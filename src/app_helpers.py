"""
Shared helpers for the Streamlit app: paths, cached loaders, artifact checks,
live inference and the What-If voucher policy.

The app never trains anything: it only reads CSV files and .joblib artifacts
produced by src/train_models.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# All paths are resolved from this file location (works from any folder).
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_PATH = DATA_DIR / "raw" / "UrbanMart_Transactions.csv"
TABLE_PATH = DATA_DIR / "processed" / "churn_modeling_table.csv"

PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#22C55E"
CHURN_COLORS = {"0": "#3B82F6", "1": "#EF4444"}

# Model name shown in the UI -> serialized file in models/
CLASSIFIER_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "ANN (MLP)": "ann_mlp.joblib",
}
REQUIRED_FILES = ["preprocessor.joblib", "kmeans.joblib", "kmeans_scaler.joblib",
                  "kmeans_segment_map.joblib", "metadata.json", *CLASSIFIER_FILES.values()]
REQUIRED_REPORTS = ["benchmark_metrics.csv", "benchmark_train_metrics.csv",
                    "confusion_matrices.csv", "roc_curves.csv", "ann_loss_curve.csv"]

# What-If policy: probabilities in [threshold - MIDDLE_BAND, threshold) are "watch".
MIDDLE_BAND = 0.10
HIGH_SPEND_QUANTILE = 0.75

RETRAIN_HINT = "Run `python src/train_models.py` from the repository root to rebuild the artifacts."


# ---------------------------------------------------------------- loaders
@st.cache_data(show_spinner="Loading transactions...")
def load_transactions():
    """Load the raw transaction file (50,000 rows)."""
    df = pd.read_csv(RAW_PATH)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    return df


@st.cache_data
def load_customer_table():
    """Load the customer level modeling table built by the training script."""
    return pd.read_csv(TABLE_PATH)


@st.cache_data
def load_report(file_name):
    """Load one CSV report from reports/."""
    return pd.read_csv(REPORTS_DIR / file_name)


@st.cache_data
def load_metadata():
    """Load models/metadata.json as a dict."""
    return json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading serialized models...")
def load_artifacts():
    """Load every .joblib artifact once per server process (no retraining)."""
    return {
        "preprocessor": joblib.load(MODELS_DIR / "preprocessor.joblib"),
        "classifiers": {name: joblib.load(MODELS_DIR / file)
                        for name, file in CLASSIFIER_FILES.items()},
        "kmeans": joblib.load(MODELS_DIR / "kmeans.joblib"),
        "kmeans_scaler": joblib.load(MODELS_DIR / "kmeans_scaler.joblib"),
        "segment_map": joblib.load(MODELS_DIR / "kmeans_segment_map.joblib"),
    }


def check_artifacts():
    """Return a list of problems (missing files or version mismatch). Empty = OK."""
    problems = [f"Missing file: models/{f}" for f in REQUIRED_FILES
                if not (MODELS_DIR / f).exists()]
    problems += [f"Missing file: reports/{f}" for f in REQUIRED_REPORTS
                 if not (REPORTS_DIR / f).exists()]
    if not TABLE_PATH.exists():
        problems.append("Missing file: data/processed/churn_modeling_table.csv")
    if not problems:
        trained_with = load_metadata()["versions"]["scikit-learn"]
        if trained_with != sklearn.__version__:
            problems.append(f"scikit-learn version mismatch: models trained with {trained_with}, "
                            f"installed {sklearn.__version__}.")
    return problems


# ---------------------------------------------------------------- inference
def make_input_row(recency, frequency, monetary, region, age_group, gender, category):
    """Build a one row DataFrame with the same columns used in training."""
    return pd.DataFrame([{
        "Recency": recency,
        "Frequency_log": np.log1p(frequency),
        "Monetary_log": np.log1p(monetary),
        "Region": region,
        "CustomerAgeGroup": age_group,
        "CustomerGender": gender,
        "PreferredCategory": category,
    }])


def predict_churn(artifacts, model_name, row):
    """Transform the row with the saved preprocessor and return P(churn)."""
    x = artifacts["preprocessor"].transform(row)
    return float(artifacts["classifiers"][model_name].predict_proba(x)[0, 1])


def predict_segment(artifacts, row):
    """Scale RFM with the K-Means scaler and return the segment name."""
    rfm = row[["Recency", "Frequency_log", "Monetary_log"]]
    cluster = int(artifacts["kmeans"].predict(artifacts["kmeans_scaler"].transform(rfm))[0])
    return artifacts["segment_map"][cluster]


def decide_policy(probability, threshold, segment, monetary, high_spend_limit):
    """
    Voucher rule used by the What-If tab. Returns (level, message).

    level is "error", "warning" or "success" (the Streamlit alert to show).
    """
    if probability >= threshold:
        if monetary >= high_spend_limit:
            return "error", ("High churn risk and high spend: deploy a STRONG re-engagement "
                             "voucher (premium discount plus personal outreach).")
        return "error", "High churn risk: deploy a re-engagement voucher."
    if probability >= threshold - MIDDLE_BAND:
        return "warning", ("Medium risk (close to the threshold): add the customer to a "
                           "watch list and send a soft reminder campaign.")
    if segment == "Champions":
        return "success", "Low risk Champion: offer the VIP voucher (loyalty reward, early access)."
    return "success", "Low risk: no voucher needed, keep the regular communication."


def format_metric(value):
    """Format a metric value for tables: 3 decimals or N/A."""
    return "N/A" if pd.isna(value) else f"{value:.3f}"

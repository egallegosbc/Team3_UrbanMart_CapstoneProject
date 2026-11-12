"""
Feature engineering and churn target builder for UrbanMart.

The raw dataset has no churn label, so we create one with a TEMPORAL split:

    - Features use ONLY transactions BEFORE the cutoff date.
    - churn_risk_flag = 1 if the customer makes NO purchase from the cutoff
      date until the last date in the data, otherwise 0.

Because the target is measured in a future window that the features never
see, Recency is a fair feature (no target leakage).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Paths are resolved from this file, so the code works from any folder.
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "UrbanMart_Transactions.csv"
MODELING_TABLE_PATH = ROOT_DIR / "data" / "processed" / "churn_modeling_table.csv"

# Chosen after testing several windows (see docs/REPORT_FACTS.md):
# cutoff 2024-10-01 gave only 4.3% churners, 2024-11-01 gives 13.1%.
DEFAULT_CUTOFF = "2024-11-01"

NUMERIC_FEATURES = ["Recency", "Frequency_log", "Monetary_log"]
CATEGORICAL_FEATURES = ["Region", "CustomerAgeGroup", "CustomerGender", "PreferredCategory"]
TARGET = "churn_risk_flag"


def load_transactions(path=RAW_PATH):
    """Read the raw transactions CSV and parse the date column."""
    df = pd.read_csv(path)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    return df


def data_quality_report(df):
    """Return a small dict with missing value and duplicate counts."""
    return {
        "rows": int(len(df)),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_transaction_ids": int(df["TransactionID"].duplicated().sum()),
    }


def most_frequent(series):
    """Return the most frequent value of a series (first one on ties)."""
    return series.mode().iloc[0]


def build_rfm(df, reference_date):
    """
    Build Recency, Frequency and Monetary per customer.

    Recency = days between the last purchase and reference_date.
    Frequency = number of unique transactions.
    Monetary = total spend.
    """
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("TransactionDate", "max"),
        Frequency=("TransactionID", "nunique"),
        Monetary=("TransactionValue", "sum"),
    )
    rfm["Recency"] = (pd.Timestamp(reference_date) - rfm["LastPurchase"]).dt.days
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    return rfm.drop(columns="LastPurchase")


def build_modeling_table(df, cutoff=DEFAULT_CUTOFF):
    """
    Build the customer level modeling table with features and the target.

    Only customers with at least one purchase before the cutoff are kept.
    """
    cutoff = pd.Timestamp(cutoff)
    history = df[df["TransactionDate"] < cutoff]
    future = df[df["TransactionDate"] >= cutoff]

    # 1) RFM features from the history only (Recency measured to the cutoff).
    table = build_rfm(history, reference_date=cutoff)

    # 2) Customer attributes: most frequent value per customer (history only).
    attributes = history.groupby("CustomerID").agg(
        Region=("Region", most_frequent),
        CustomerAgeGroup=("CustomerAgeGroup", most_frequent),
        CustomerGender=("CustomerGender", most_frequent),
        PreferredCategory=("ProductCategory", most_frequent),
    )
    table = table.join(attributes)

    # 3) Target: 1 if the customer did NOT buy in the future window.
    active_after_cutoff = set(future["CustomerID"])
    table[TARGET] = (~table.index.isin(active_after_cutoff)).astype(int)

    return table.reset_index()


def window_info(df, cutoff=DEFAULT_CUTOFF):
    """Describe the observation window used for the target."""
    cutoff = pd.Timestamp(cutoff)
    last_date = df["TransactionDate"].max()
    return {
        "cutoff_date": str(cutoff.date()),
        "last_date": str(last_date.date()),
        "window_days": int((last_date - cutoff).days + 1),
    }


if __name__ == "__main__":
    transactions = load_transactions()
    print("Data quality:", data_quality_report(transactions))
    modeling = build_modeling_table(transactions)
    MODELING_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(MODELING_TABLE_PATH, index=False)
    print("Window:", window_info(transactions))
    print("Customers:", len(modeling), "| churn rate:", round(modeling[TARGET].mean(), 4))
    print("Saved:", MODELING_TABLE_PATH)

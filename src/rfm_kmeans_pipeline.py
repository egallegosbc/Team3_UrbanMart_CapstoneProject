from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

RAW_PATH = PROJECT_ROOT / os.getenv(
    "DATA_RAW_PATH",
    "data/raw/UrbanMart_Transactions.csv",
)

PROCESSED_DIR = PROJECT_ROOT / os.getenv(
    "DATA_PROCESSED_DIR",
    "data/processed",
)

REPORTS_DIR = PROJECT_ROOT / os.getenv(
    "REPORTS_DIR",
    "reports",
)

RANDOM_STATE = int(
    os.getenv("RANDOM_STATE", "42")
)

K_MIN = int(
    os.getenv("K_MIN", "2")
)

K_MAX = int(
    os.getenv("K_MAX", "10")
)


REQUIRED_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "TransactionDate",
    "TransactionValue",
    "ProductCategory",
    "PaymentMethod",
    "CustomerGender",
    "CustomerAgeGroup",
    "Region",
]


SEGMENT_ORDER = [
    "Champions",
    "Developing",
    "At Risk",
]


SEGMENT_ACTIONS = {
    "Champions": (
        "VIP rewards, early access, referrals"
    ),
    "Developing": (
        "Cross-sell / upsell bundles"
    ),
    "At Risk": (
        "Win-back offers, urgency discount"
    ),
}


SEGMENT_COLORS = {
    "Developing": "#3b82f6",
    "Champions": "#10b981",
    "At Risk": "#f59e0b",
}


# ============================================================
# 2. VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame) -> None:
    """
    Validate that the input dataset contains all expected fields
    and does not contain critical data problems.
    """

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "The CSV file is empty."
        )

    if df["CustomerID"].isna().any():
        raise ValueError(
            "CustomerID contains missing values."
        )

    if df["TransactionID"].duplicated().any():
        duplicates = int(
            df["TransactionID"].duplicated().sum()
        )

        raise ValueError(
            f"TransactionID contains "
            f"{duplicates} duplicate values."
        )

    if df["TransactionDate"].isna().any():
        raise ValueError(
            "TransactionDate contains invalid "
            "or missing dates."
        )

    if df["TransactionValue"].isna().any():
        raise ValueError(
            "TransactionValue contains missing values."
        )

    if (df["TransactionValue"] < 0).any():
        raise ValueError(
            "TransactionValue contains negative values."
        )


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Read the UrbanMart transaction dataset and print
    ingestion evidence for the report.
    """

    print()
    print("=" * 65)
    print("STEP 1: DATA INGESTION")
    print("=" * 65)

    print(
        f"File loaded from: {RAW_PATH}"
    )

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found at:\n{RAW_PATH}\n\n"
            "Check DATA_RAW_PATH inside .env"
        )

    df = pd.read_csv(RAW_PATH)

    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"],
        errors="coerce",
    )

    validate_input(df)

    print(
        f"Shape (rows, columns): {df.shape}"
    )

    print(
        f"Total rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique customers: "
        f"{df['CustomerID'].nunique():,}"
    )

    print(
        f"Unique transactions: "
        f"{df['TransactionID'].nunique():,}"
    )

    print(
        "Date range: "
        f"{df['TransactionDate'].min().date()} "
        "to "
        f"{df['TransactionDate'].max().date()}"
    )

    total_missing = int(
        df.isna().sum().sum()
    )

    print(
        f"Missing values: "
        f"{total_missing} total"
    )

    print(
        "Columns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    return df


# ============================================================
# 4. FEATURE ENGINEERING: RFM
# ============================================================

def build_rfm(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Recency, Frequency and Monetary features.

    Recency:
        Number of days since customer's last transaction.

    Frequency:
        Number of transactions for each customer.

    Monetary:
        Total historical customer spending.
    """

    snapshot_date = (
        df["TransactionDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby(
            "CustomerID",
            as_index=False,
        )
        .agg(
            LastPurchase=(
                "TransactionDate",
                "max",
            ),
            Frequency=(
                "TransactionID",
                "count",
            ),
            Monetary=(
                "TransactionValue",
                "sum",
            ),
        )
    )

    rfm["Recency"] = (
        snapshot_date
        - rfm["LastPurchase"]
    ).dt.days

    rfm = rfm[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ]

    return rfm


# ============================================================
# 5. TRANSFORM AND SCALE
# ============================================================

def transform_and_scale(
    rfm: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Apply log1p to Frequency and Monetary and then
    StandardScaler to all three RFM variables.
    """

    features = rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ].copy()

    # Frequency and Monetary are right-skewed.
    features["Frequency"] = np.log1p(
        features["Frequency"]
    )

    features["Monetary"] = np.log1p(
        features["Monetary"]
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    return scaled_features, scaler


# ============================================================
# 6. SELECT OPTIMAL NUMBER OF CLUSTERS
# ============================================================

def select_k(
    scaled_features: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """
    Test k = 2 ... 10 using:
      - Inertia (Elbow Method)
      - Silhouette Score
    """

    print()
    print("=" * 65)
    print("STEP 2: SELECTING OPTIMAL K")
    print("=" * 65)

    results = []

    for k in range(
        K_MIN,
        K_MAX + 1,
    ):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(
            scaled_features
        )

        silhouette = silhouette_score(
            scaled_features,
            labels,
        )

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
            }
        )

        print(
            f"k={k:>2} | "
            f"inertia={model.inertia_:,.4f} | "
            f"silhouette={silhouette:.4f}"
        )

    metrics = pd.DataFrame(
        results
    )

    best_index = (
        metrics["silhouette_score"]
        .idxmax()
    )

    best_k = int(
        metrics.loc[
            best_index,
            "k",
        ]
    )

    print()
    print(
        f"Selected k = {best_k}"
    )

    print(
        "(highest Silhouette Score)"
    )

    return metrics, best_k


# ============================================================
# 7. ELBOW AND SILHOUETTE CHARTS
# ============================================================

def plot_k_selection(
    metrics: pd.DataFrame,
    best_k: int,
) -> None:

    # Elbow chart
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        metrics["k"],
        metrics["inertia"],
        marker="o",
    )

    plt.axvline(
        best_k,
        linestyle="--",
        label=f"Chosen k={best_k}",
    )

    plt.title(
        "Elbow Method - "
        "Inertia vs. Number of Clusters (k)"
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Inertia "
        "(within-cluster sum of squares)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "elbow_method.png",
        dpi=200,
    )

    plt.close()

    # Silhouette chart
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        metrics["k"],
        metrics["silhouette_score"],
        marker="o",
    )

    plt.axvline(
        best_k,
        linestyle="--",
        label=f"Chosen k={best_k}",
    )

    plt.title(
        "Silhouette Score vs. "
        "Number of Clusters (k)"
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Average Silhouette Score"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "silhouette_scores.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# 8. BUSINESS LABELING
# ============================================================

def label_clusters(
    rfm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Translate anonymous K-Means clusters into useful
    business labels.

    First:
        Highest average Recency = At Risk

    Then:
        Rank remaining clusters using standardized
        engagement:

        - Recency + Frequency + Monetary

    where lower Recency is better.
    """

    profiles = (
        rfm.groupby(
            "Cluster",
            as_index=False,
        )
        .agg(
            Customers=(
                "CustomerID",
                "count",
            ),
            AvgRecency=(
                "Recency",
                "mean",
            ),
            AvgFrequency=(
                "Frequency",
                "mean",
            ),
            AvgMonetary=(
                "Monetary",
                "mean",
            ),
            TotalRevenue=(
                "Monetary",
                "sum",
            ),
        )
    )

    profile_features = profiles[
        [
            "AvgRecency",
            "AvgFrequency",
            "AvgMonetary",
        ]
    ].copy()

    means = (
        profile_features.mean()
    )

    stds = (
        profile_features
        .std(ddof=0)
        .replace(0, 1)
    )

    zscores = (
        profile_features - means
    ) / stds

    profiles["EngagementScore"] = (
        -zscores["AvgRecency"]
        + zscores["AvgFrequency"]
        + zscores["AvgMonetary"]
    )

    # Highest Recency = longest time without purchase
    at_risk_cluster = int(
        profiles.loc[
            profiles["AvgRecency"].idxmax(),
            "Cluster",
        ]
    )

    remaining = profiles[
        profiles["Cluster"]
        != at_risk_cluster
    ]

    champions_cluster = int(
        remaining.loc[
            remaining[
                "EngagementScore"
            ].idxmax(),
            "Cluster",
        ]
    )

    developing_cluster = int(
        remaining.loc[
            remaining["Cluster"]
            != champions_cluster,
            "Cluster",
        ].iloc[0]
    )

    cluster_labels = {
        champions_cluster:
            "Champions",

        developing_cluster:
            "Developing",

        at_risk_cluster:
            "At Risk",
    }

    rfm["Segment"] = (
        rfm["Cluster"]
        .map(cluster_labels)
    )

    profiles["Segment"] = (
        profiles["Cluster"]
        .map(cluster_labels)
    )

    return rfm, profiles


# ============================================================
# 9. SEGMENT SUMMARY
# ============================================================

def build_segment_summary(
    rfm: pd.DataFrame,
) -> pd.DataFrame:

    total_customers = len(
        rfm
    )

    total_revenue = (
        rfm["Monetary"].sum()
    )

    summary = (
        rfm.groupby(
            "Segment",
            as_index=False,
        )
        .agg(
            Customers=(
                "CustomerID",
                "count",
            ),
            AvgRecency=(
                "Recency",
                "mean",
            ),
            AvgFrequency=(
                "Frequency",
                "mean",
            ),
            AvgMonetary=(
                "Monetary",
                "mean",
            ),
            TotalRevenue=(
                "Monetary",
                "sum",
            ),
        )
    )

    summary[
        "CustomerSharePct"
    ] = (
        summary["Customers"]
        / total_customers
        * 100
    )

    summary[
        "RevenueSharePct"
    ] = (
        summary["TotalRevenue"]
        / total_revenue
        * 100
    )

    summary[
        "SuggestedAction"
    ] = (
        summary["Segment"]
        .map(SEGMENT_ACTIONS)
    )

    summary[
        "MonetaryLevel"
    ] = (
        summary["Segment"]
        .map(
            {
                "Champions": "High",
                "Developing": "Med.",
                "At Risk": "Med.",
            }
        )
    )

    summary["Segment"] = pd.Categorical(
        summary["Segment"],
        categories=SEGMENT_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("Segment")
        .reset_index(drop=True)
    )

    summary["Segment"] = (
        summary["Segment"]
        .astype("object")
    )

    return summary


# ============================================================
# 10. CUSTOMER SEGMENT SCATTER PLOT
# ============================================================

def plot_segments(
    rfm: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(9, 6)
    )

    for segment in SEGMENT_ORDER:

        segment_data = rfm[
            rfm["Segment"]
            == segment
        ]

        bubble_size = np.clip(
            segment_data["Frequency"] * 4,
            20,
            180,
        )

        plt.scatter(
            segment_data["Recency"],
            segment_data["Monetary"],
            s=bubble_size,
            alpha=0.65,
            label=segment,
            color=SEGMENT_COLORS[
                segment
            ],
            edgecolors="none",
        )

    plt.title(
        "Customer Segments - RFM (K-Means)\n"
        "Bubble size = Frequency"
    )

    plt.xlabel(
        "Recency "
        "(days since last purchase)"
    )

    plt.ylabel(
        "Monetary "
        "(total spend)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "rfm_segments_scatter.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# 11. SAVE OUTPUT FILES
# ============================================================

def save_outputs(
    rfm: pd.DataFrame,
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:

    # Simple CRM-ready output:
    # CustomerID -> Segment
    (
        rfm[
            [
                "CustomerID",
                "Segment",
            ]
        ]
        .sort_values(
            "CustomerID"
        )
        .to_csv(
            PROCESSED_DIR
            / "customer_segments.csv",
            index=False,
        )
    )

    # Complete RFM details
    (
        rfm
        .sort_values(
            "CustomerID"
        )
        .to_csv(
            PROCESSED_DIR
            / "rfm_customer_details.csv",
            index=False,
        )
    )

    profiles.to_csv(
        REPORTS_DIR
        / "cluster_profiles.csv",
        index=False,
    )

    summary.to_csv(
        REPORTS_DIR
        / "segment_summary.csv",
        index=False,
    )

    metrics.to_csv(
        REPORTS_DIR
        / "k_selection_metrics.csv",
        index=False,
    )


# ============================================================
# 12. TERMINAL RESULTS
# ============================================================

def print_final_results(
    final_silhouette: float,
    summary: pd.DataFrame,
) -> None:

    print()
    print("=" * 65)
    print("STEP 4: FINAL K-MEANS MODEL")
    print("=" * 65)

    print(
        f"Final Silhouette Score: "
        f"{final_silhouette:.4f}"
    )

    print()
    print("Segment results:")

    total_customers = int(
        summary["Customers"].sum()
    )

    for _, row in summary.iterrows():

        customers = int(
            row["Customers"]
        )

        percentage = (
            Decimal(customers)
            * Decimal("100")
            / Decimal(total_customers)
        ).quantize(
            Decimal("0.1"),
            rounding=ROUND_HALF_UP,
        )

        print(
            f"- {row['Segment']}: "
            f"{customers:,} customers "
            f"({percentage}%) | "
            f"Avg R="
            f"{row['AvgRecency']:.1f}d | "
            f"Avg F="
            f"{row['AvgFrequency']:.1f} | "
            f"Avg M="
            f"{row['AvgMonetary']:,.0f} | "
            f"Revenue share="
            f"{row['RevenueSharePct']:.1f}%"
        )

    print()
    print("Files created:")

    print(
        "- data/processed/"
        "customer_segments.csv"
    )

    print(
        "- data/processed/"
        "rfm_customer_details.csv"
    )

    print(
        "- reports/"
        "k_selection_metrics.csv"
    )

    print(
        "- reports/"
        "cluster_profiles.csv"
    )

    print(
        "- reports/"
        "segment_summary.csv"
    )

    print(
        "- reports/"
        "elbow_method.png"
    )

    print(
        "- reports/"
        "silhouette_scores.png"
    )

    print(
        "- reports/"
        "rfm_segments_scatter.png"
    )


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main() -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------
    # Data ingestion
    # ---------------------------------------

    df = load_data()

    # ---------------------------------------
    # RFM
    # ---------------------------------------

    print()
    print("=" * 65)
    print("STEP 2A: FEATURE ENGINEERING - RFM")
    print("=" * 65)

    rfm = build_rfm(
        df
    )

    print(
        f"RFM table created: "
        f"{len(rfm):,} customers"
    )

    # ---------------------------------------
    # Transform
    # ---------------------------------------

    print()
    print("=" * 65)
    print("STEP 2B: TRANSFORM & SCALE")
    print("=" * 65)

    scaled_features, _ = (
        transform_and_scale(
            rfm
        )
    )

    print(
        "Applied log1p to "
        "Frequency and Monetary."
    )

    print(
        "Applied StandardScaler to "
        "Recency, log(Frequency), "
        "log(Monetary)."
    )

    # ---------------------------------------
    # Select k
    # ---------------------------------------

    metrics, best_k = (
        select_k(
            scaled_features
        )
    )

    plot_k_selection(
        metrics,
        best_k,
    )

    # ---------------------------------------
    # Final K-Means
    # ---------------------------------------

    print()
    print("=" * 65)
    print("STEP 3: FITTING FINAL K-MEANS")
    print("=" * 65)

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    rfm["Cluster"] = (
        final_model.fit_predict(
            scaled_features
        )
    )

    final_silhouette = (
        silhouette_score(
            scaled_features,
            rfm["Cluster"],
        )
    )

    # ---------------------------------------
    # Business labels
    # ---------------------------------------

    rfm, profiles = (
        label_clusters(
            rfm
        )
    )

    summary = (
        build_segment_summary(
            rfm
        )
    )

    # ---------------------------------------
    # Visualizations
    # ---------------------------------------

    plot_segments(
        rfm
    )

    # ---------------------------------------
    # Save everything
    # ---------------------------------------

    save_outputs(
        rfm,
        profiles,
        summary,
        metrics,
    )

    # ---------------------------------------
    # Final terminal output
    # ---------------------------------------

    print_final_results(
        final_silhouette,
        summary,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# URBANMART CUSTOMER INTELLIGENCE
# RFM-BASED CUSTOMER SEGMENTATION WITH K-MEANS
# Team 3 - Data Mining & Modern AI Systems
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
file_path = "data/raw/UrbanMart_Transactions.csv"
df = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("STEP 1 - DATA INGESTION")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\nColumns:")
for column in df.columns:
    print(f" - {column}")
print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# STEP 2 - DATA PREPARATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 2 - DATA PREPARATION")
print("=" * 70)

# Convert TransactionDate to datetime
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
print(f"Date range: {df['TransactionDate'].min().date()} "
      f"to {df['TransactionDate'].max().date()}")
print(f"Unique customers: {df['CustomerID'].nunique():,}")

# ============================================================
# STEP 3 - RFM FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3 - RFM FEATURE ENGINEERING")
print("=" * 70)

# Reference date = one day after the last transaction
reference_date = df["TransactionDate"].max() + pd.Timedelta(days=1)
print(f"Reference date: {reference_date.date()}")

# Create RFM table
rfm = (
    df.groupby("CustomerID")
    .agg(
        Recency=(
            "TransactionDate",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=(
            "TransactionID",
            "nunique"
        ),
        Monetary=(
            "TransactionValue",
            "sum"
        )
    )
    .reset_index()
)

print("\nRFM table created successfully.")
print(f"Customers in RFM table: {len(rfm):,}")
print("\nFirst 5 customers:")
print(rfm.head())
print("\nRFM descriptive statistics:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# ============================================================
# STEP 4 - LOG TRANSFORMATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 4 - LOG TRANSFORMATION")
print("=" * 70)

# Frequency and Monetary are right-skewed,
# so log1p is used to stabilize their distributions.
rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
print("log1p transformation applied to:")
print(" - Frequency")
print(" - Monetary")

# ============================================================
# STEP 5 - STANDARDIZATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 5 - FEATURE SCALING")
print("=" * 70)

features = [
    "Recency",
    "Frequency_log",
    "Monetary_log"
]
X = rfm[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("StandardScaler applied successfully.")
print("\nFeatures used for K-Means:")
for feature in features:
    print(f" - {feature}")

# ============================================================
# STEP 6 - TEST K = 2 TO 10
# ============================================================
print("\n" + "=" * 70)
print("STEP 6 - K-MEANS MODEL COMPARISON")
print("=" * 70)

k_values = range(2, 11)
inertia_values = []
silhouette_values = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(
        X_scaled,
        cluster_labels
    )
    inertia_values.append(inertia)
    silhouette_values.append(silhouette)
    print(
        f"k = {k} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )

# ============================================================
# STEP 7 - IDENTIFY BEST K
# ============================================================
print("\n" + "=" * 70)
print("STEP 7 - MODEL SELECTION")
print("=" * 70)

best_index = np.argmax(silhouette_values)
best_k = list(k_values)[best_index]
best_silhouette = silhouette_values[best_index]
print(f"Best k based on Silhouette Score: {best_k}")
print(f"Best Silhouette Score: {best_silhouette:.4f}")

# For the UrbanMart project, k = 3 was selected.
chosen_k = 3
print(f"\nFinal selected number of clusters: k = {chosen_k}")

# ============================================================
# STEP 8 - DIAGNOSTIC PLOTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 8 - GENERATING DIAGNOSTIC CHARTS")
print("=" * 70)

os.makedirs(
    "reports/figures",
    exist_ok=True
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# ------------------------------
# Inertia / Elbow Plot
# ------------------------------
axes[0].plot(
    list(k_values),
    inertia_values,
    marker="o"
)
axes[0].axvline(
    x=chosen_k,
    linestyle="--"
)
axes[0].set_title(
    "K-Means Elbow Method"
)
axes[0].set_xlabel(
    "Number of Clusters (k)"
)
axes[0].set_ylabel(
    "Inertia"
)
axes[0].grid(
    alpha=0.3
)

# ------------------------------
# Silhouette Plot
# ------------------------------
axes[1].plot(
    list(k_values),
    silhouette_values,
    marker="o"
)
axes[1].axvline(
    x=chosen_k,
    linestyle="--"
)
axes[1].set_title(
    "Silhouette Score by k"
)
axes[1].set_xlabel(
    "Number of Clusters (k)"
)
axes[1].set_ylabel(
    "Silhouette Score"
)
axes[1].grid(
    alpha=0.3
)

plt.tight_layout()
figure_path = (
    "reports/figures/rfm_diagnostics.png"
)
plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print(
    f"Diagnostic figure saved to: "
    f"{figure_path}"
)

# ============================================================
# STEP 9 - FINAL K-MEANS MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 9 - FINAL K-MEANS MODEL")
print("=" * 70)

final_model = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)
rfm["Cluster"] = final_model.fit_predict(
    X_scaled
)
final_silhouette = silhouette_score(
    X_scaled,
    rfm["Cluster"]
)
print(
    f"Final Silhouette Score: "
    f"{final_silhouette:.4f}"
)
print("\nCustomers per cluster:")
print(
    rfm["Cluster"]
    .value_counts()
    .sort_index()
)

# ============================================================
# STEP 10 - RAW CLUSTER PROFILES
# ============================================================
print("\n" + "=" * 70)
print("STEP 10 - CLUSTER PROFILES")
print("=" * 70)

cluster_profile = (
    rfm.groupby("Cluster")
    .agg(
        Customers=(
            "CustomerID",
            "count"
        ),
        Avg_Recency=(
            "Recency",
            "mean"
        ),
        Avg_Frequency=(
            "Frequency",
            "mean"
        ),
        Avg_Monetary=(
            "Monetary",
            "mean"
        ),
        Total_Revenue=(
            "Monetary",
            "sum"
        )
    )
)
total_revenue = (
    cluster_profile["Total_Revenue"].sum()
)
cluster_profile["Revenue_Share_Pct"] = (
    cluster_profile["Total_Revenue"]
    / total_revenue
    * 100
)
print("\nRaw cluster profile:")
print(
    cluster_profile.round(2)
)

# ============================================================
# STEP 11 - Z-SCORE CLUSTER INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 11 - CLUSTER INTERPRETATION")
print("=" * 70)

profile_for_zscore = cluster_profile[
    [
        "Avg_Recency",
        "Avg_Frequency",
        "Avg_Monetary"
    ]
]
profile_mean = profile_for_zscore.mean()
profile_std = profile_for_zscore.std(ddof=0)
cluster_zscores = (
    profile_for_zscore - profile_mean
) / profile_std
cluster_zscores.columns = [
    "Recency_Z",
    "Frequency_Z",
    "Monetary_Z"
]
cluster_profile = cluster_profile.join(
    cluster_zscores
)

# Higher Recency means the customer has not
# purchased for a longer period of time.
# Therefore, the cluster with the highest
# average Recency becomes "At Risk".
at_risk_cluster = (
    cluster_profile["Avg_Recency"].idxmax()
)

# Remaining clusters
remaining_clusters = [
    cluster
    for cluster in cluster_profile.index
    if cluster != at_risk_cluster
]

# Engagement score:
# lower Recency is better,
# higher Frequency is better,
# higher Monetary is better.
cluster_profile["Engagement_Score"] = (
    -cluster_profile["Recency_Z"]
    + cluster_profile["Frequency_Z"]
    + cluster_profile["Monetary_Z"]
)
champion_cluster = (
    cluster_profile
    .loc[remaining_clusters, "Engagement_Score"]
    .idxmax()
)
developing_cluster = [
    cluster
    for cluster in remaining_clusters
    if cluster != champion_cluster
][0]

segment_map = {
    champion_cluster: "Champions",
    developing_cluster: "Developing",
    at_risk_cluster: "At Risk"
}
rfm["Segment"] = (
    rfm["Cluster"]
    .map(segment_map)
)
cluster_profile["Segment"] = (
    cluster_profile.index
    .map(segment_map)
)

# ============================================================
# STEP 12 - FINAL SEGMENT SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("STEP 12 - FINAL CUSTOMER SEGMENTS")
print("=" * 70)

segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=(
            "CustomerID",
            "count"
        ),
        Avg_Recency=(
            "Recency",
            "mean"
        ),
        Avg_Frequency=(
            "Frequency",
            "mean"
        ),
        Avg_Monetary=(
            "Monetary",
            "mean"
        ),
        Revenue=(
            "Monetary",
            "sum"
        )
    )
)
segment_summary[
    "Customer_Share_Pct"
] = (
    segment_summary["Customers"]
    / len(rfm)
    * 100
)
segment_summary[
    "Revenue_Share_Pct"
] = (
    segment_summary["Revenue"]
    / segment_summary["Revenue"].sum()
    * 100
)
segment_order = [
    "Champions",
    "Developing",
    "At Risk"
]
segment_summary = (
    segment_summary
    .reindex(segment_order)
)
print(
    segment_summary.round(2)
)

# ============================================================
# STEP 13 - BUSINESS INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 13 - BUSINESS INTERPRETATION")
print("=" * 70)

print(
    "\nCHAMPIONS:"
    "\nRecent customers with high purchasing "
    "frequency and strong monetary value."
    "\nRecommended action: loyalty, VIP benefits, "
    "early access and referral programs."
)
print(
    "\nDEVELOPING:"
    "\nActive customers with lower frequency or "
    "spending than Champions."
    "\nRecommended action: cross-sell, bundles, "
    "recommendations and next-order incentives."
)
print(
    "\nAT RISK:"
    "\nCustomers who previously purchased but have "
    "gone significantly longer without buying."
    "\nRecommended action: win-back campaigns, "
    "email/SMS reminders and targeted offers."
)

# ============================================================
# STEP 14 - WIN-BACK ALERT
# ============================================================
print("\n" + "=" * 70)
print("STEP 14 - WIN-BACK ALERT")
print("=" * 70)

winback_threshold = 60
rfm["Winback_Alert"] = np.where(
    rfm["Recency"] > winback_threshold,
    "Review",
    "No Alert"
)
customers_flagged = (
    rfm["Winback_Alert"]
    .eq("Review")
    .sum()
)
print(
    f"Win-back threshold: "
    f"Recency > {winback_threshold} days"
)
print(
    f"Customers flagged for review: "
    f"{customers_flagged}"
)

# ============================================================
# STEP 15 - SAVE OUTPUT
# ============================================================
print("\n" + "=" * 70)
print("STEP 15 - SAVE RESULTS")
print("=" * 70)

os.makedirs(
    "reports",
    exist_ok=True
)
output_path = (
    "reports/urbanmart_customer_segments.csv"
)
rfm.to_csv(
    output_path,
    index=False
)
print(
    f"Customer segmentation file saved to: "
    f"{output_path}"
)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("URBANMART RFM PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
print(
    f"Transactions analyzed: "
    f"{len(df):,}"
)
print(
    f"Unique customers: "
    f"{len(rfm):,}"
)
print(
    f"Selected clusters: "
    f"{chosen_k}"
)
print(
    f"Final Silhouette Score: "
    f"{final_silhouette:.4f}"
)
print("\nFinal segment sizes:")
for segment in segment_order:
    count = (
        rfm["Segment"]
        .eq(segment)
        .sum()
    )
    percentage = (
        count / len(rfm) * 100
    )
    print(
        f"{segment}: "
        f"{count:,} customers "
        f"({percentage:.1f}%)"
    )
print("\nPipeline finished.")

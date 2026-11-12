"""
Training script for the UrbanMart churn models (MLOps pipeline).

Run from any folder with:   python src/train_models.py

What it does:
    1. Builds the customer modeling table (temporal churn target).
    2. Splits train/test 80/20 (stratified, random_state=42).
    3. Fits the preprocessor (imputer + scaler + one-hot) on TRAIN only.
    4. Trains Logistic Regression, Decision Tree and an ANN (MLPClassifier).
    5. Reproduces the Milestone 2 RFM K-Means model (k=3).
    6. Saves every model with joblib in models/ and all test metrics in reports/,
       so the Streamlit app only READS files and never retrains.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Make "src" importable when the script is run as a file from any folder.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import (CATEGORICAL_FEATURES, DEFAULT_CUTOFF, MODELING_TABLE_PATH,  # noqa: E402
                          NUMERIC_FEATURES, TARGET, build_modeling_table, build_rfm,
                          data_quality_report, load_transactions, window_info)

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
RANDOM_STATE = 42
KMEANS_FEATURES = ["Recency", "Frequency_log", "Monetary_log"]


def build_preprocessor():
    """Numeric: median imputer + StandardScaler. Categorical: imputer + one-hot."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def build_classifiers():
    """Return the three classifiers with simple, documented hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE),
        # MLPClassifier has no class_weight option: it is trained on an oversampled
        # (balanced) copy of the train set instead. alpha=1.0 is a strong L2 penalty
        # chosen to reduce overfitting; with it the optimizer converges (< 500 iterations).
        "ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=1.0,
            max_iter=500, random_state=RANDOM_STATE),
    }


def oversample_minority(x, y):
    """Duplicate random churners until both classes have the same size (train only)."""
    y = np.asarray(y)
    rng = np.random.default_rng(RANDOM_STATE)
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    extra = rng.choice(positives, len(negatives), replace=True)
    index = np.concatenate([negatives, extra])
    return x[index], y[index]


def classification_metrics(y_true, y_pred, y_proba):
    """Compute the benchmark metrics for one classifier."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        # R-squared of the predicted probability versus the 0/1 label.
        "R2 (probability vs label)": r2_score(y_true, y_proba),
        "Silhouette": np.nan,
    }


def name_segments(rfm):
    """
    Milestone 2 naming rule: highest average Recency = At Risk; among the other
    two clusters, the highest engagement score (-Recency + Frequency + Monetary,
    z-scores of the cluster means) = Champions; the last one = Developing.
    """
    profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    z = (profile - profile.mean()) / profile.std(ddof=0)
    engagement = -z["Recency"] + z["Frequency"] + z["Monetary"]
    at_risk = int(profile["Recency"].idxmax())
    champions = int(engagement.drop(at_risk).idxmax())
    developing = [int(c) for c in profile.index if c not in (at_risk, champions)][0]
    return {champions: "Champions", developing: "Developing", at_risk: "At Risk"}


def train_kmeans(transactions):
    """Reproduce Milestone 2: K-Means k=3 on full history RFM. Returns model parts."""
    reference_date = transactions["TransactionDate"].max() + pd.Timedelta(days=1)
    rfm = build_rfm(transactions, reference_date)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(rfm[KMEANS_FEATURES])
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(x_scaled)
    segment_map = name_segments(rfm)
    rfm["Segment"] = rfm["Cluster"].map(segment_map)
    silhouette = silhouette_score(x_scaled, rfm["Cluster"])
    return kmeans, scaler, segment_map, rfm, silhouette


def main():
    """Run the full training pipeline and save artifacts and reports."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1) Data and target
    transactions = load_transactions()
    quality = data_quality_report(transactions)
    table = build_modeling_table(transactions, DEFAULT_CUTOFF)
    table.to_csv(MODELING_TABLE_PATH, index=False)

    # 2) Stratified train/test split
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, x_test, y_train, y_test = train_test_split(
        table[features], table[TARGET], test_size=0.2,
        stratify=table[TARGET], random_state=RANDOM_STATE)

    # 3) Preprocessor fitted on TRAIN only
    preprocessor = build_preprocessor()
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # 4) Classifiers: train, save, evaluate on TEST (and on train for overfitting check)
    file_names = {"Logistic Regression": "logistic_regression.joblib",
                  "Decision Tree": "decision_tree.joblib", "ANN (MLP)": "ann_mlp.joblib"}
    test_rows, train_rows, cm_rows, roc_rows = [], [], [], []
    for name, model in build_classifiers().items():
        if name == "ANN (MLP)":
            model.fit(*oversample_minority(x_train_t, y_train))
        else:
            model.fit(x_train_t, y_train)
        joblib.dump(model, MODELS_DIR / file_names[name])

        proba = model.predict_proba(x_test_t)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_rows.append({"Model": name, **classification_metrics(y_test, pred, proba)})

        train_proba = model.predict_proba(x_train_t)[:, 1]
        train_rows.append({"Model": name, **classification_metrics(
            y_train, (train_proba >= 0.5).astype(int), train_proba)})

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        cm_rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_rows += [{"Model": name, "FPR": f, "TPR": t} for f, t in zip(fpr, tpr)]

    ann = joblib.load(MODELS_DIR / "ann_mlp.joblib")
    pd.DataFrame({"Iteration": range(1, len(ann.loss_curve_) + 1),
                  "Loss": ann.loss_curve_}).to_csv(REPORTS_DIR / "ann_loss_curve.csv", index=False)

    # 5) K-Means (Milestone 2 reproduction)
    kmeans, km_scaler, segment_map, rfm_full, silhouette = train_kmeans(transactions)
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    joblib.dump(km_scaler, MODELS_DIR / "kmeans_scaler.joblib")
    joblib.dump(segment_map, MODELS_DIR / "kmeans_segment_map.joblib")
    rfm_full.reset_index().to_csv(REPORTS_DIR / "kmeans_full_history_segments.csv", index=False)
    sizes = rfm_full["Segment"].value_counts().to_dict()

    # Optional K-Means churn score: apply the segment rule to the TEST customers'
    # pre-cutoff features and treat "At Risk" as a churn prediction.
    test_clusters = kmeans.predict(km_scaler.transform(x_test[KMEANS_FEATURES]))
    km_pred = np.array([segment_map[c] == "At Risk" for c in test_clusters]).astype(int)
    km_row = {"Model": "K-Means (At Risk = churn)",
              "Accuracy": accuracy_score(y_test, km_pred),
              "Precision": precision_score(y_test, km_pred, zero_division=0),
              "Recall": recall_score(y_test, km_pred), "F1": f1_score(y_test, km_pred),
              "ROC-AUC": np.nan, "R2 (probability vs label)": np.nan, "Silhouette": silhouette}
    tn, fp, fn, tp = confusion_matrix(y_test, km_pred).ravel()
    cm_rows.append({"Model": km_row["Model"], "TN": tn, "FP": fp, "FN": fn, "TP": tp})

    pd.DataFrame([km_row] + test_rows).to_csv(REPORTS_DIR / "benchmark_metrics.csv", index=False)
    pd.DataFrame(train_rows).to_csv(REPORTS_DIR / "benchmark_train_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(REPORTS_DIR / "confusion_matrices.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(REPORTS_DIR / "roc_curves.csv", index=False)

    # 6) Metadata for reproducibility and version checks in the app
    metadata = {
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_STATE,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "joblib": joblib.__version__},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "transformed_feature_names": list(preprocessor.get_feature_names_out()),
        "kmeans_features": KMEANS_FEATURES,
        "target": TARGET,
        "window": window_info(transactions, DEFAULT_CUTOFF),
        "data_quality": quality,
        "class_balance": {"customers": int(len(table)), "churn_positive": int(table[TARGET].sum()),
                          "churn_rate": round(float(table[TARGET].mean()), 4)},
        "split": {"train": int(len(x_train)), "test": int(len(x_test)), "stratified": True},
        "ann": {"hidden_layers": [32, 16], "iterations": int(ann.n_iter_),
                "final_loss": round(float(ann.loss_curve_[-1]), 4)},
        "kmeans": {"k": 3, "silhouette": round(float(silhouette), 4),
                   "segment_map": {str(k): v for k, v in segment_map.items()},
                   "segment_sizes": {k: int(v) for k, v in sizes.items()}},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 7) Short readable summary
    print("Data quality:", quality)
    print("Window:", metadata["window"], "| class balance:", metadata["class_balance"])
    print(f"K-Means silhouette: {silhouette:.4f} | segment sizes: {sizes}")
    print(f"ANN iterations: {ann.n_iter_} | final loss: {ann.loss_curve_[-1]:.4f}")
    print("\nTest set metrics:")
    print(pd.DataFrame([km_row] + test_rows).set_index("Model").round(3).to_string())
    print("\nTrain set metrics (overfitting check):")
    print(pd.DataFrame(train_rows).set_index("Model").round(3).to_string())


if __name__ == "__main__":
    main()

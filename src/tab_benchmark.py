"""Tab 3 - Multi-Model Benchmark Suite (all metrics read from reports/, test set only)."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_helpers as helpers

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                  "R2 (probability vs label)", "Silhouette"]


def confusion_figure(row):
    """Plotly heatmap for one row of reports/confusion_matrices.csv."""
    matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Greens",
                    x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                    template=helpers.PLOT_TEMPLATE, title=row["Model"])
    fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def roc_figure(roc_points, benchmark):
    """Overlay of the ROC curves of the three classifiers plus the random line."""
    fig = go.Figure()
    for model, points in roc_points.groupby("Model", sort=False):
        auc = benchmark.loc[benchmark["Model"] == model, "ROC-AUC"].iloc[0]
        fig.add_trace(go.Scatter(x=points["FPR"], y=points["TPR"], mode="lines",
                                 name=f"{model} (AUC {auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC 0.5)",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(template=helpers.PLOT_TEMPLATE, height=420, title="ROC curves (test set)",
                      xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def render_benchmark_tab():
    """Benchmark table, confusion matrices, ROC curves and ANN loss curve."""
    benchmark = helpers.load_report("benchmark_metrics.csv")
    train = helpers.load_report("benchmark_train_metrics.csv")
    matrices = helpers.load_report("confusion_matrices.csv")
    roc_points = helpers.load_report("roc_curves.csv")
    loss = helpers.load_report("ann_loss_curve.csv")
    metadata = helpers.load_metadata()
    split = metadata["split"]

    st.success(f"All metrics below are computed on the **held-out test set** "
               f"({split['test']} customers, stratified 20%). Models were trained on "
               f"{split['train']} customers. Nothing is recomputed in the app.")

    # 1) Comparative table
    st.subheader("Benchmark table (test set)")
    shown = benchmark.set_index("Model")[METRIC_COLUMNS]
    st.dataframe(shown.map(helpers.format_metric), width="stretch")
    st.caption("R2 is computed on the predicted probability versus the true 0/1 label. "
               "K-Means is unsupervised: its Silhouette (0.383) is on the full history RFM; "
               "its classification metrics treat the 'At Risk' cluster as a churn prediction.")

    classifiers = benchmark.dropna(subset=["ROC-AUC"])
    best = classifiers.loc[classifiers["ROC-AUC"].idxmax()]
    best_recall = classifiers.loc[classifiers["Recall"].idxmax()]
    st.markdown(
        f"**Interpretation.** {best['Model']} has the best ROC-AUC ({best['ROC-AUC']:.3f}) and "
        f"{best_recall['Model']} the highest recall ({best_recall['Recall']:.3f}). "
        "However, every ROC-AUC is close to 0.5: with a leakage-free temporal target, purchase "
        "history in this dataset barely predicts who stops buying in the next 60 days. "
        "Transactions are spread almost uniformly over time, so this is an honest result, not "
        "a bug. Using class balancing, the models trade accuracy for recall: they catch about "
        "half of the churners at the cost of many false alarms (low precision). For a CRM "
        "manager this means vouchers should be cheap, because many targeted customers would "
        "not have churned anyway.")

    with st.expander("Train versus test metrics (overfitting check)"):
        st.dataframe(train.set_index("Model")[METRIC_COLUMNS[:6]].map(helpers.format_metric),
                     width="stretch")
        st.caption("Logistic Regression and the Decision Tree have similar train and test "
                   "scores (no overfitting, but also little signal). The ANN scores much "
                   "higher on train than on test: it memorizes noise (overfitting).")

    # 2) Confusion matrices side by side
    st.subheader("Confusion matrices (test set, threshold 0.5)")
    columns = st.columns(len(matrices))
    for column, (_, row) in zip(columns, matrices.iterrows()):
        column.plotly_chart(confusion_figure(row), width="stretch",
                            config={"displayModeBar": False})

    # 3) ROC curves
    st.plotly_chart(roc_figure(roc_points, benchmark), width="stretch")

    # 4) ANN loss curve
    st.subheader("ANN (MLP) training loss convergence")
    fig = px.line(loss, x="Iteration", y="Loss", template=helpers.PLOT_TEMPLATE,
                  color_discrete_sequence=[helpers.ACCENT])
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")
    ann = metadata["ann"]
    ann_train = train.set_index("Model").loc["ANN (MLP)", "ROC-AUC"]
    ann_test = benchmark.set_index("Model").loc["ANN (MLP)", "ROC-AUC"]
    st.markdown(
        f"**Interpretation.** The loss (log-loss on the balanced training data) falls quickly "
        f"in the first iterations and then flattens; the optimizer stopped after "
        f"{ann['iterations']} iterations at a loss of {ann['final_loss']}, meaning it converged. "
        f"A low training loss does not mean a good model: train ROC-AUC is {ann_train:.3f} but "
        f"test ROC-AUC is {ann_test:.3f}. The network (layers {ann['hidden_layers']}, strong L2 "
        "penalty alpha=1.0) still overfits the noise, which is why the simpler Logistic "
        "Regression generalizes at least as well.")

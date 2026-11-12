"""
Smoke test for the Streamlit app using streamlit.testing.v1.AppTest.

Run with either:
    python -m pytest tests/smoke_test.py -v
    python tests/smoke_test.py
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = ROOT_DIR / "app.py"
sys.path.insert(0, str(ROOT_DIR))

from src.app_helpers import decide_policy  # noqa: E402

# Scenarios chosen from the Logistic Regression coefficients (see docs/REPORT_FACTS.md).
LOW_RISK = {"wi_recency": 120, "wi_frequency": 15, "wi_monetary": 30_000_000,
            "wi_region": "Yogyakarta", "wi_age": "36-50", "wi_gender": "Male",
            "wi_category": "Sport"}
HIGH_RISK = {"wi_recency": 0, "wi_frequency": 40, "wi_monetary": 100_000_000,
             "wi_region": "Medan", "wi_age": "18-25", "wi_gender": "Female",
             "wi_category": "Fashion"}


def start_app():
    """Run the app once and fail if any exception was raised."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def set_scenario(at, values):
    """Set the What-If widgets to the given values and rerun the app."""
    for key, value in values.items():
        widget = at.slider(key=key) if key in ("wi_recency", "wi_frequency",
                                               "wi_monetary") else at.selectbox(key=key)
        widget.set_value(value)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def churn_probability(at):
    """Read the churn probability metric shown in the What-If tab."""
    return next(m.value for m in at.metric if m.label == "Churn probability")


def policy_messages(at):
    """Return the texts of every error/warning/success alert on the page."""
    return {"error": [e.value for e in at.error], "warning": [w.value for w in at.warning],
            "success": [s.value for s in at.success]}


def test_app_runs_with_four_tabs_and_kpis():
    """The app starts, shows 4 tabs and the KPI cards."""
    at = start_app()
    assert len(at.tabs) == 4
    labels = [m.label for m in at.metric]
    for kpi in ["Customers", "Transactions", "Churn rate (target)", "Best model (ROC-AUC)"]:
        assert kpi in labels


def test_transaction_summary_tables_do_not_mix_types():
    """Tab 1: numeric and categorical/date summaries are separate, with no 1970 dates."""
    at = start_app()
    tables = [frame.value for frame in at.dataframe]
    numeric = next(t for t in tables if "TransactionValue" in t.index and "mean" in t.columns)
    assert list(numeric.index) == ["TransactionValue"]
    assert all(dtype.kind == "f" for dtype in numeric.dtypes)
    assert numeric.loc["TransactionValue", "max"] > 1_000_000

    categorical = next(t for t in tables if "TransactionDate" in t.index and "top" in t.columns)
    assert "TransactionValue" not in categorical.index
    assert categorical.loc["TransactionDate", "unique"] == 730
    for table in (numeric, categorical):
        assert "1970-01-01" not in table.astype(str).to_string()


def test_low_and_high_risk_scenarios():
    """Changing sliders updates the prediction and the alert type."""
    at = start_app()
    at = set_scenario(at, LOW_RISK)
    low_probability = churn_probability(at)
    assert any("Low risk" in text for text in policy_messages(at)["success"])

    at = set_scenario(at, HIGH_RISK)
    high_probability = churn_probability(at)
    assert high_probability != low_probability
    assert any("re-engagement voucher" in text for text in policy_messages(at)["error"])

    # The history table must have logged both scenarios.
    assert len(at.session_state["wi_history"]) >= 2


def test_every_model_and_presets_run():
    """Each serialized classifier and both preset buttons work without errors."""
    at = start_app()
    for model in ["Logistic Regression", "Decision Tree", "ANN (MLP)"]:
        at.selectbox(key="wi_model").set_value(model).run()
        assert not at.exception
    for button in at.button:
        if button.label.startswith("Preset"):
            button.click().run()
            assert not at.exception


def test_policy_rules():
    """Pure unit test of the voucher rule."""
    assert decide_policy(0.70, 0.5, "Developing", 1e8, 6.7e7)[1].startswith("High churn risk and")
    assert decide_policy(0.70, 0.5, "Developing", 1e7, 6.7e7)[0] == "error"
    assert decide_policy(0.45, 0.5, "Developing", 1e7, 6.7e7)[0] == "warning"
    assert "VIP" in decide_policy(0.10, 0.5, "Champions", 1e7, 6.7e7)[1]


if __name__ == "__main__":
    tests = [test_app_runs_with_four_tabs_and_kpis,
             test_transaction_summary_tables_do_not_mix_types, test_low_and_high_risk_scenarios,
             test_every_model_and_presets_run, test_policy_rules]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All smoke tests passed.")

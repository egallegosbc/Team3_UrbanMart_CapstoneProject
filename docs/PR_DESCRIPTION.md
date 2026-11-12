# feat: Streamlit capstone dashboard with serialized MLOps pipeline

## Objective

Milestone 3 (Lab Activity 23): a live Streamlit dashboard for the Retention and CRM Manager that
loads a serialized scikit-learn pipeline (K-Means, Logistic Regression, Decision Tree, ANN) to
predict `churn_risk_flag` and simulate VIP / re-engagement voucher decisions.

## What changed (per file)

- `requirements.txt`: rewritten as UTF-8 with direct dependencies; adds streamlit 1.67.0, plotly 7.1.0, pytest.
- `.streamlit/config.toml`: custom dark theme (green accent).
- `src/features.py`: customer feature table and temporal churn target (cutoff 2024-11-01, 60-day window, 13.05% churn).
- `src/train_models.py`: stratified 80/20 split, train-only preprocessor, 3 classifiers + Milestone 2 K-Means; saves `models/*.joblib`, `models/metadata.json` and test-set reports.
- `data/processed/churn_modeling_table.csv`, `models/`, `reports/*` (new files only; Milestone 2 files unchanged).
- `app.py`: page config, artifact/version check, KPI cards, 4 tabs.
- `src/app_helpers.py`: pathlib paths, `@st.cache_data` / `@st.cache_resource` loaders, inference and voucher policy.
- `src/tab_eda.py`, `src/tab_preprocessing.py`, `src/tab_benchmark.py`, `src/tab_whatif.py`: one module per tab.
- `tests/smoke_test.py`: AppTest smoke test.
- `docs/`, `README.md`: report facts, AI log draft, defense notes, updated README.

## How to run (Windows PowerShell)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
python src\train_models.py        # optional, artifacts are committed
python -m streamlit run app.py
python -m pytest tests\smoke_test.py -v
```

## How it was tested

- `tests/smoke_test.py`: 4 passed (app starts with 4 tabs and KPIs; low-risk scenario -> success,
  high-risk scenario -> error; every model and both presets run; voucher rules unit test).
- Headless server: `/_stcore/health` returned `ok`.
- Launched from a different working directory: no path errors.
- Fresh clone of this branch + new venv + `pip install -r requirements.txt`: pip check OK, 4 tests passed.
- Models are only loaded with `joblib.load` (no `fit` in the app); first run about 4 s.

## Results (held-out test set, 400 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Silhouette |
|---|---|---|---|---|---|---|
| K-Means (At Risk = churn) | 0.735 | 0.065 | 0.077 | 0.070 | N/A | 0.383 |
| Logistic Regression | 0.565 | 0.153 | 0.519 | 0.237 | 0.538 | N/A |
| Decision Tree | 0.425 | 0.129 | 0.596 | 0.212 | 0.523 | N/A |
| ANN (MLP 32-16) | 0.708 | 0.158 | 0.288 | 0.204 | 0.530 | N/A |

K-Means matches Milestone 2 (silhouette 0.383, 996 / 729 / 275). Classifiers are near random because
the target is leakage-free and the synthetic data has almost no temporal signal (documented).

## Checklist

- [x] Exactly 4 tabs, dark theme, KPI cards, cache decorators, session_state
- [x] Models and scalers loaded from .joblib at runtime, version check against metadata.json
- [x] Benchmark on held-out test set: table, confusion matrices, ROC curves, ANN loss curve + interpretation
- [x] What-If sliders with data-driven ranges, presets via callbacks, scenario history
- [x] Paths resolved with pathlib from the file location
- [x] Smoke test passing, fresh clone test passing
- [x] Milestone 2 code and outputs unchanged
- [ ] Teammate review
- [ ] Merge into main (after review and team approval)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

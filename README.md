# UrbanMart Customer Intelligence

Team 3 capstone project (Data Mining and Modern AI Systems, Universidad Anahuac Mayab).
Client persona: **Retention and CRM Manager**. Target: **churn_risk_flag**.

- Milestone 2: RFM customer segmentation with K-Means (k=3: Champions, Developing, At Risk).
- Milestone 3: live Streamlit dashboard with a serialized MLOps pipeline (K-Means, Logistic
  Regression, Decision Tree and an ANN) and a real-time What-If voucher simulator.

## Dataset

UrbanMart Transactions 2023-2024: 50,000 transactions, 2,000 customers, 9 columns,
dates 2023-01-01 to 2024-12-30. No missing values and no duplicates.

## Churn target (no leakage)

The data has no churn label, so it is built with a temporal split:

- Features (Recency, log Frequency, log Monetary, Region, Age group, Gender, Preferred category)
  use only transactions **before 2024-11-01**.
- `churn_risk_flag = 1` if the customer makes **no purchase in the next 60 days**
  (2024-11-01 to 2024-12-30). Churn rate: 13.05% (261 of 2,000).

## Project structure

```
app.py                     Streamlit dashboard (KPIs + 4 tabs)
.streamlit/config.toml     dark theme
requirements.txt           direct dependencies (Python 3.14)
data/raw/                  UrbanMart_Transactions.csv
data/processed/            Milestone 2 outputs + churn_modeling_table.csv
models/                    *.joblib artifacts + metadata.json
reports/                   Milestone 2 figures + test-set benchmark files
src/features.py            customer features and churn target
src/train_models.py        training, serialization and evaluation
src/app_helpers.py         cached loaders, inference and voucher policy
src/tab_*.py               one module per dashboard tab
src/rfm_kmeans_pipeline.py Milestone 2 pipeline (unchanged)
tests/smoke_test.py        Streamlit AppTest smoke test
docs/                      report facts, AI log draft, defense notes
```

## Installation (Windows, PowerShell)

```powershell
cd C:\path\to\repo
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
```

## Train the models (optional, artifacts are already committed)

```powershell
python src\train_models.py
```

## Run the dashboard

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. Tabs: Data Explorer and EDA, Preprocessing and Correlation,
Multi-Model Benchmark, What-If Inference.

## Run the tests

```powershell
python -m pytest tests\smoke_test.py -v
```

## Results (held-out test set, 400 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Silhouette |
|---|---|---|---|---|---|---|
| K-Means (At Risk = churn) | 0.735 | 0.065 | 0.077 | 0.070 | N/A | 0.383 |
| Logistic Regression | 0.565 | 0.153 | 0.519 | 0.237 | 0.538 | N/A |
| Decision Tree | 0.425 | 0.129 | 0.596 | 0.212 | 0.523 | N/A |
| ANN (MLP 32-16) | 0.708 | 0.158 | 0.288 | 0.204 | 0.530 | N/A |

K-Means reproduces Milestone 2 exactly (silhouette 0.383, segments 996 / 729 / 275).
All classifiers are close to random (ROC-AUC about 0.53): in this synthetic dataset purchase
history barely predicts future inactivity once leakage is removed. The ANN overfits
(train ROC-AUC 0.914 vs test 0.530). See `docs/REPORT_FACTS.md` for full details.

# Report Facts - UrbanMart Customer Intelligence (Milestone 3)

All numbers below come from `python src/train_models.py` (random seed 42) and are read by the app
from `reports/` and `models/`. Classification metrics are on the **held-out test set** (400 customers).

## 1. Final file tree (files added or changed in this milestone are marked *)

```
repo/
├── app.py                          * Streamlit entry point (KPIs + 4 tabs)
├── requirements.txt                * UTF-8, direct dependencies only
├── README.md                       *
├── .streamlit/config.toml          * dark theme
├── data/
│   ├── raw/UrbanMart_Transactions.csv
│   └── processed/
│       ├── customer_segments.csv            (Milestone 2)
│       ├── rfm_customer_details.csv         (Milestone 2)
│       └── churn_modeling_table.csv       * customer table with churn_risk_flag
├── models/                         * serialized pipeline
│   ├── preprocessor.joblib, logistic_regression.joblib, decision_tree.joblib, ann_mlp.joblib
│   ├── kmeans.joblib, kmeans_scaler.joblib, kmeans_segment_map.joblib
│   └── metadata.json
├── reports/
│   ├── (Milestone 2 files unchanged)
│   ├── benchmark_metrics.csv, benchmark_train_metrics.csv          *
│   ├── confusion_matrices.csv, roc_curves.csv, ann_loss_curve.csv   *
│   └── kmeans_full_history_segments.csv                             *
├── src/
│   ├── rfm_kmeans_pipeline.py       (Milestone 2, unchanged)
│   ├── features.py                * feature builder + temporal target
│   ├── train_models.py            * training + serialization + evaluation
│   ├── app_helpers.py             * cached loaders, inference, voucher policy
│   ├── tab_eda.py, tab_preprocessing.py, tab_benchmark.py, tab_whatif.py  *
├── tests/smoke_test.py            * AppTest smoke test
└── docs/                          * REPORT_FACTS.md, AI_LOG_DRAFT.md, DEFENSE_NOTES.md
```

## 2. Dataset facts

- 50,000 transactions, 2,000 customers, 9 columns (TransactionID, CustomerID, TransactionDate,
  TransactionValue, ProductCategory, PaymentMethod, CustomerGender, CustomerAgeGroup, Region).
- Dates 2023-01-01 to 2024-12-30. Total revenue 126.7 billion (local currency).
- Missing values: **0**. Duplicate rows: **0**. Duplicate TransactionIDs: **0**.
  Treatment: median imputer (numeric) and most-frequent imputer (categorical) kept in the pipeline.
- Transactions are spread almost uniformly over time (about 6,200 per quarter in all 8 quarters).

## 3. Target definition (churn_risk_flag)

- Cutoff date: **2024-11-01**. Future window: 2024-11-01 to 2024-12-30 = **60 days**.
- Features use only transactions **before** the cutoff (Recency is measured to the cutoff).
- `churn_risk_flag = 1` if the customer has **no** transaction in the 60-day window, else 0.
- Customers with no purchase before the cutoff are excluded (none in practice: all 2,000 kept).
- Class balance: **261 churners of 2,000 = 13.05%**.

Windows tested (positive rate must be roughly 8%-45%):

| Cutoff | Window | Churn rate | Decision |
|---|---|---|---|
| 2024-09-01 | 121 days | 1.7% | too rare |
| 2024-10-01 (default) | 91 days | 4.3% | too rare |
| 2024-10-15 | 77 days | 7.2% | borderline |
| **2024-11-01** | **60 days** | **13.1%** | **chosen** |
| 2024-11-15 | 46 days | 20.8% | workable, but window is very short |

Univariate ROC-AUC of each RFM feature against the target was between 0.48 and 0.54 for every
window, so no window gives a strongly predictable target (see section 12).

## 4. Features and preprocessing

- Numeric: Recency, Frequency_log = log1p(Frequency), Monetary_log = log1p(Monetary)
  -> median imputer -> StandardScaler.
- Categorical (most frequent value per customer): Region, CustomerAgeGroup, CustomerGender,
  PreferredCategory -> most-frequent imputer -> OneHotEncoder(handle_unknown="ignore").
- Final model input: 3 numeric + 17 one-hot = **20 columns**.
- Split 80/20 stratified, random_state=42: 1,600 train / 400 test. Preprocessor fit on train only.

## 5. Models and hyperparameters

| Model | Settings |
|---|---|
| Logistic Regression | class_weight="balanced", max_iter=1000 |
| Decision Tree | max_depth=4, min_samples_leaf=20, class_weight="balanced" |
| ANN (MLPClassifier) | hidden layers (32, 16), relu, adam, alpha=1.0, max_iter=500; trained on a randomly oversampled (balanced) copy of the train set because MLP has no class_weight |
| K-Means | k=3, random_state=42, n_init=10, own StandardScaler on full-history Recency, log1p Frequency, log1p Monetary |

## 6. Final metrics (TEST set, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | R2 (prob. vs label) | Silhouette |
|---|---|---|---|---|---|---|---|
| K-Means (At Risk = churn) | 0.735 | 0.065 | 0.077 | 0.070 | N/A | N/A | 0.383 |
| **Logistic Regression** | 0.565 | 0.153 | 0.519 | **0.237** | **0.538** | -1.154 | N/A |
| Decision Tree | 0.425 | 0.129 | **0.596** | 0.212 | 0.523 | -1.179 | N/A |
| ANN (MLP) | **0.708** | **0.158** | 0.288 | 0.204 | 0.530 | -0.724 | N/A |

Train set (overfitting check): LR ROC-AUC 0.589, Decision Tree 0.585, **ANN 0.914** (test 0.530 -> overfitting).

R2 is negative because the balanced models output probabilities around 0.5 while 87% of labels are 0;
it is reported because the rubric asks for it, but it is not a good metric for classification.

## 7. Confusion matrices (test set)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 199 | 149 | 25 | 27 |
| Decision Tree | 139 | 209 | 21 | 31 |
| ANN (MLP) | 268 | 80 | 37 | 15 |
| K-Means (At Risk = churn) | 290 | 58 | 48 | 4 |

Test set: 348 non-churners, 52 churners.

## 8. ANN and K-Means details

- ANN: 2 hidden layers (32 and 16 neurons, ReLU), output sigmoid. Converged after **317 iterations**,
  final training loss **0.5259** (starts at 0.851). No ConvergenceWarning.
- K-Means: k=3, silhouette **0.3827** (Milestone 2: about 0.383 -> reproduced).
  Segment sizes **Champions 996, Developing 729, At Risk 275** (identical to Milestone 2).
  Cluster ids: 1 = Champions, 0 = Developing, 2 = At Risk.

## 9. What-If rules (src/app_helpers.py, `decide_policy`)

- P(churn) >= threshold (default 0.50) -> `st.error`: re-engagement voucher.
  If Monetary >= 67,100,885 (75th percentile of customer spend) -> STRONG voucher (premium discount + personal outreach).
- threshold - 0.10 <= P < threshold -> `st.warning`: watch list + soft reminder.
- Otherwise `st.success`: low risk; if the K-Means segment is Champions -> VIP voucher.
- Slider ranges from the customer table: Recency 0-200 days, Frequency min-max (max 40),
  Monetary rounded to 100,000 steps up to about 109.8 million.
- Presets use the median RFM and the most frequent categories of the customers in each K-Means segment.

## 10. Demo scenarios (Logistic Regression, threshold 0.50) - real outputs

| Scenario | Recency | Freq | Monetary | Region | Age | Gender | Category | Segment | P(churn) | Alert |
|---|---|---|---|---|---|---|---|---|---|---|
| Low risk VIP | 5 | 35 | 90,000,000 | Yogyakarta | 36-50 | Male | Sport | Champions | 0.316 | success: VIP voucher |
| Medium | 20 | 25 | 60,000,000 | Surabaya | 26-35 | Female | Electronics | Champions | 0.466 | warning: watch list |
| High risk | 0 | 40 | 100,000,000 | Medan | 18-25 | Female | Fashion | Champions | 0.688 | error: STRONG re-engagement voucher |
| Low (disagreement) | 120 | 15 | 30,000,000 | Yogyakarta | 36-50 | Male | Sport | At Risk | 0.220 | success: no voucher |
| Preset At Risk | 77 | 22 | 54,400,000 | Medan | 26-35 | Female | Beauty | At Risk | 0.585 | error: re-engagement voucher |
| Preset Champion | 16 | 27 | 70,900,000 | Surabaya | 26-35 | Female | Beauty | Champions | 0.580 | error: STRONG voucher |

Other models for the high-risk scenario: Decision Tree 0.554, ANN 0.349.

## 11. Commands

```bash
python src/train_models.py            # rebuild table, models, reports (about 10 s)
python -m streamlit run app.py        # start the dashboard (http://localhost:8501)
python -m pytest tests/smoke_test.py -v
```

Verification done: smoke test **5/5 passed** on main (the 5th test was added with the summary
table fix, problem 9); headless server health check `ok`; launched from
`C:\Users\romed` (different folder) without path errors; joblib loading about 3.6 s, first full
app run about 4 s; fresh clone of the pushed branch + new venv + `pip install -r requirements.txt`
+ `pip check` (no broken requirements) + smoke test: **4 passed** (run before the 5th test existed).

## 12. Problems found and how they were fixed

1. **Default window too imbalanced**: cutoff 2024-10-01 gave 4.3% churners -> moved to 2024-11-01 (13.1%).
2. **No predictive signal in the data**: all ROC-AUC values are about 0.52-0.54. The dataset is
   synthetic and uniform over time. We kept the leakage-free target and report this honestly
   instead of defining churn from Recency (which would give a fake ROC-AUC near 1.0).
3. **ANN first version** (alpha 0.001, no balancing): did not converge in 500 iterations, train
   ROC-AUC 0.97 vs test 0.53, almost never predicted churn. Fixed with oversampling of the
   minority class and alpha=1.0 -> converges in 317 iterations; overfitting is reduced but still visible.
4. **Logistic Regression learns noise**: the Recency coefficient is slightly negative (-0.03), so in
   the What-If tab a higher Recency slightly LOWERS churn probability. Categorical features
   (PreferredCategory, age group) have the largest coefficients.
5. **Streamlit deprecations**: `use_container_width` replaced by `width="stretch"`.
6. **KPI cards truncated** on narrow screens -> 2 rows of 3 cards. Plotly toolbar covered
   confusion matrix titles -> toolbar hidden for those charts.
7. **requirements.txt was UTF-16** -> rewritten as UTF-8 with direct dependencies only.
8. **Widget key conflicts**: preset buttons use `on_click` callbacks that update `st.session_state`
   before widgets are created; widgets do not pass `value=` when their key already has a value.
9. **Summary table showed 1970-01-01 dates** (found by the team during review, after PR #2 was
   merged): in the Data Explorer tab, `describe(include="all")` put timestamps and numbers in the
   same columns, so Streamlit rendered the TransactionValue mean, min, quartiles and max as
   1970-01-01 dates. Fixed by showing two tables: a numeric-only `describe` (TransactionValue)
   and a count / unique / top / freq table for text and date columns, with dates converted to
   `YYYY-MM-DD` text. A new AppTest check fails on the old code and passes on the fix. Merged in PR #3.

## 13. Git

- Branch: `feature/capstone-streamlit` (from origin/main f011b60). Pushed, no force push.
- `rfm-kmeans-pipeline-parte-2` was left untouched (on the remote, 1 commit ahead of main, not merged).
- Pull requests (opened manually from the web, GitHub CLI is not installed on this machine):
  - PR #2 https://github.com/egallegosbc/Team3_UrbanMart_CapstoneProject/pull/2 - dashboard,
    pipeline, tests and docs. Approved by a teammate and **merged into main** (merge commit a595431).
  - PR #3 https://github.com/egallegosbc/Team3_UrbanMart_CapstoneProject/pull/3 - summary table
    fix (problem 9). **Merged into main** (merge commit 2a1b00b).
  - A third PR from the same branch adds this documentation update.

Commits (oldest last):

```
docs: record summary table fix and PR #2 / PR #3 in report and AI log
fix: split transaction summary into numeric and categorical/date tables
docs: add pull request URL to report facts
docs: add report facts, AI log draft, defense notes, PR description and updated README
test: add AppTest smoke test for tabs, What-If alerts and policy rules
fix: improve KPI card layout and chart readability on laptop screens
feat: add Streamlit app shell with dark theme, KPI cards and four tabs
feat: add real-time What-If inference tab with presets and scenario history
feat: add Multi-Model Benchmark tab with test-set metrics, ROC and loss curve
feat: add Preprocessing and Correlation tab
feat: add Data Explorer and EDA tab
feat: add cached loaders, artifact checks and inference helpers for the app
feat: add training script with serialized joblib models and test-set reports
feat: add customer feature builder and temporal churn_risk_flag target
chore: add Streamlit dark theme config and clean UTF-8 requirements
```

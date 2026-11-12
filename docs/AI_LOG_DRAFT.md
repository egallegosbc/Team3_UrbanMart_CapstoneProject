# Responsible AI Pair Programming Log (draft)

## 1. Tool and model

- Tool: Claude Code (Claude desktop app, Code tab) on Windows 11.
- Model: Claude Opus 5.5 (`claude-opus-5-5`).
- Date of session: 2026-11-12.
- Human in the loop: Team 3 student (repository work done on branch `feature/capstone-streamlit`).

## 2. Key AI suggestions and layout decisions

- **Temporal target instead of a Recency rule** to avoid leakage; tested 5 cutoffs and chose
  2024-11-01 (60-day window, 13.05% churners) because the default cutoff gave only 4.3%.
- **Separate preprocessor and models** (`preprocessor.joblib` + one file per model) so the app can
  show the scaler effect and the one-hot matrix using the same saved objects used for inference.
- **ANN without class_weight**: suggested random oversampling of churners on the train set only,
  and a strong L2 penalty (alpha=1.0) after the first ANN did not converge and overfitted.
- **Training once, app only reads**: all metrics, confusion matrices, ROC points and the loss curve
  are written to `reports/` by `src/train_models.py`; the app never calls `fit`.
- **One module per tab** (`src/tab_*.py`) plus `src/app_helpers.py` so every file is short.
- **Presets with `on_click` callbacks** that write to `st.session_state` before widgets render.
- **Artifact and version check** (`check_artifacts`) with a clear `st.error` and `st.stop`.
- **Honest reporting** of near-random ROC-AUC (about 0.53) instead of tuning the target to look good.

## 3. Checks the human should repeat and verify personally

1. **Scaler alignment**: in `train_models.py` confirm `preprocessor.fit_transform(x_train)` and only
   `transform` on `x_test`; in the app confirm the What-If row goes through the same saved preprocessor.
2. **K-Means scaler**: the What-If segment uses `kmeans_scaler.joblib`, not the classifier preprocessor.
3. **Path handling**: run `python -m streamlit run C:\full\path\to\app.py` from another folder.
4. **session_state keys**: every What-If widget key starts with `wi_`; no widget passes `value=`.
5. **Metrics on test set**: open `reports/benchmark_metrics.csv` and compare with
   `benchmark_train_metrics.csv`; recompute one confusion matrix by hand (TN+FP = 348, FN+TP = 52).
6. **No leakage**: in `features.py`, `history = df[df["TransactionDate"] < cutoff]` is the only
   data used for features; the target uses only `future`.
7. **Reproducibility**: delete `models/` and rerun `python src/train_models.py`; results must match.
8. **Milestone 2 match**: silhouette 0.3827 and segment sizes 996 / 729 / 275.

## 4. Limitations and bugs found during testing

- **No predictive signal**: ROC-AUC 0.52-0.54 for all classifiers; data is synthetic and uniform.
- **ANN overfitting**: train ROC-AUC 0.914 vs test 0.530 even with alpha=1.0.
- **Counterintuitive coefficient**: Recency has a small negative coefficient in Logistic Regression.
- **Streamlit reruns**: every widget change reruns the whole script; heavy work is cached with
  `@st.cache_data` / `@st.cache_resource`. The scenario history only appends when the scenario changes.
- **Widget key conflicts**: avoided by setting defaults with `st.session_state.setdefault` before
  widgets and by using callbacks for presets.
- **Path errors**: Milestone 2 script uses paths relative to the working directory; new code uses
  `Path(__file__).resolve()`. Verified by launching from `C:\Users\romed`.
- **Version issues**: Python 3.14.7 worked with every pinned wheel (no need for Python 3.12);
  `use_container_width` is deprecated in Streamlit 1.67 and was replaced by `width="stretch"`.
- **requirements.txt encoding**: the original file was UTF-16; rewritten as UTF-8.
- **No GitHub CLI** on the machine: the pull request must be opened from the web compare page.

## 5. Verbatim prompt given to the AI

The full prompt is reproduced below exactly as it was given.

```text
ROLE AND LANGUAGE

You are my pair programmer for a university capstone milestone. Talk to me in Spanish. Write all code, comments, UI text, commit messages and documentation in English. I am a student and I must defend this project live and from memory, so the code must be simple, readable and well commented. No over engineering. Every file should be understandable by a student in one reading. Keep each file under about 300 lines and give every function a short docstring.

Work only inside this repository folder: C:\Users\romed\Documents\UrbanMart_Team3\repo
Do not read, edit or delete anything outside it. Ignore the parent folder (it only has my old .bat and .txt scratch files).

CONTEXT

University: Universidad Anahuac Mayab. Course: Data Mining and Modern AI Systems (IIND4417, period 202660), Professor Dr. Victor Cardoso.
Task: Laboratory Activity 23, Capstone Progress Review 3 of 3, Milestone 3: live interactive Streamlit AI dashboard with a serialized MLOps pipeline.
Team 3. Project: UrbanMart Customer Intelligence. Client persona: Retention and CRM Manager.
Target variable: churn_risk_flag. Required models: RFM K-Means, Logistic Regression, ANN (neural network).
Required What-If scenario: adjust recency and monetary spend to deploy VIP or re-engagement vouchers.
GitHub remote: https://github.com/egallegosbc/Team3_UrbanMart_CapstoneProject.git (repo owner is a teammate, my GitHub user is rmeouchi-max).

What the professor requires (summary of the rubric):
1. Live Streamlit app with exactly 4 tabs: Data Explorer and EDA, Preprocessing and Correlation Matrix, Multi-Model Benchmark Suite, Real-Time What-If Scenario Inference Engine. Custom dark theme, @st.cache_data for data loading, st.session_state for reactive state, KPI metric cards at the top, responsive What-If sliders. It must run locally without crashes.
2. The app must dynamically load pretrained .joblib model files and scaler objects (no hardcoded predictions). Benchmark table comparing classical models vs the neural network with Accuracy, Precision, Recall, F1, ROC-AUC, R-squared, Silhouette. Also side by side confusion matrices, ROC curves, and the ANN loss convergence curve with a written interpretation. Metrics must come from a held out test set, never only from training data.
3. GitHub repo with standard layout: app.py, /data, /models, /.streamlit, requirements.txt. Work on branch feature/capstone-streamlit, conventional commit messages, a pull request with a real description that a teammate reviews, then merged into main.
4. Live defense: 8 minutes demo plus 4 minutes Q&A, from memory. (This part I do myself.)

VERIFIED FACTS ABOUT MY MACHINE AND REPO (checked before writing this prompt)

1. Windows, user romed. Python 3.14.7 through the py launcher. Git 2.55 installed. winget is available. Conda is NOT installed. Node/npx is NOT on PATH (so do not rely on Playwright or any browser automation).
2. The repo at repo\ was cloned from the remote above. Local branches: main and rfm-kmeans-pipeline-parte-2 (currently checked out, with amended commits). An earlier push from this machine failed with 403 permission denied, and later a remote ref origin/rfm-kmeans-pipeline-parte-2 appeared, so the push may have worked after. Verify, do not assume.
3. Existing files in repo\: src\rfm_kmeans_pipeline.py (Milestone 2 pipeline, uses paths relative to the current working directory), src\__init__.py, README.md, requirements.txt, .gitignore, .env.example, notebooks\.gitkeep, data\raw\UrbanMart_Transactions.csv, data\processed\customer_segments.csv, data\processed\rfm_customer_details.csv, reports\ (cluster_profiles.csv, segment_summary.csv, k_selection_metrics.csv, elbow_method.png, silhouette_scores.png, rfm_segments_scatter.png).
4. requirements.txt is saved as UTF-16 (looks garbled in some tools). It pins: pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, matplotlib 3.11.2, python-dotenv 1.2.3 and their dependencies. It has no streamlit or plotly yet.
5. Dataset: 50,000 transactions, 2,000 customers, 9 columns: TransactionID, CustomerID, TransactionDate, TransactionValue, ProductCategory, PaymentMethod, CustomerGender, CustomerAgeGroup, Region. Dates from 2023-01-01 to 2024-12-30. Values are large (customer total spend is in the tens of millions, local currency, Indonesian cities). There is NO churn_risk_flag column. You must create it (see step 3).
6. Milestone 2 results that must be preserved and reproduced: K-Means with k=3 on Recency, log1p(Frequency), log1p(Monetary), StandardScaler, random_state=42, n_init=10. Silhouette about 0.383. Segments: Champions 996 customers, Developing 729, At Risk 275. Segment naming rule is in src\rfm_kmeans_pipeline.py (At Risk = highest average recency, Champions = highest engagement score among the rest).

GROUND RULES

1. Never force push. Never rewrite history that was already pushed. Never commit directly to main. Never commit .env or secrets.
2. Do not delete or change the behavior of src\rfm_kmeans_pipeline.py or the existing files in reports\ and data\. New code goes in new files.
3. Do not merge a pull request until I explicitly tell you to.
4. Set PYTHONUTF8=1 when running Python on this machine (Windows console encoding breaks on non ASCII output). Do not print emojis from scripts.
5. Resolve every path with pathlib from the file location (Path(__file__).resolve().parent), so the app works no matter which folder it is launched from. Path errors are an explicit grading penalty.
6. No TensorFlow or PyTorch (Python 3.14 compatibility risk). The ANN is scikit-learn MLPClassifier.
7. If a decision is truly blocking, ask me one short question in Spanish. Otherwise decide, keep going, and record the decision.
8. Academic integrity: code must be authentic and understandable by me. Do not copy code from other repositories.

STEP 0. INSPECT BEFORE TOUCHING ANYTHING

Run: git status, git branch -a, git remote -v, git log --oneline --all --graph -15, git fetch --all --prune, gh auth status (if gh exists). Read all existing files listed above. Tell me in Spanish what state the repo is in: is the parte-2 branch on the remote, is it merged into main, are there uncommitted changes. Do not discard my work on rfm-kmeans-pipeline-parte-2; leave that branch alone.

STEP 1. GIT SETUP

1. Base: latest origin/main. Create branch feature/capstone-streamlit from it and push it with upstream tracking. If the push returns 403, stop and tell me (I probably need a collaborator invite); do not fork silently.
2. If the updated src\rfm_kmeans_pipeline.py or other needed files exist only on rfm-kmeans-pipeline-parte-2 and not on main, tell me and ask before bringing them over.
3. Teammates may push to the same branch. Always git pull --rebase before pushing, never force.
4. Commit in small logical steps with conventional messages (feat:, fix:, chore:, docs:, test:). Planned commits: chore (streamlit config and requirements), feat (features and target builder), feat (training script and joblib artifacts), feat (app shell, theme, KPI cards, cached loaders), feat (tab 1), feat (tab 2), feat (tab 3), feat (tab 4), test (smoke test), docs (README, docs folder).

STEP 2. ENVIRONMENT

1. Create a virtual environment .venv in the repo root (it is already in .gitignore) with py -3.14 -m venv .venv. Activate it and install.
2. Rewrite requirements.txt as UTF-8 (no BOM). Keep the existing pins that are used, add streamlit and plotly (and any other direct dependency you really import). Prefer a short clean list of direct dependencies over a full pip freeze. After installing, verify with pip check.
3. If a dependency has no wheel for Python 3.14, install Python 3.12 with winget (winget install Python.Python.3.12), recreate .venv with it, and tell me which version ended up being used.
4. Create .streamlit\config.toml with a dark theme: base = "dark", dark background, one accent color, readable font. Use the plotly_dark template in charts.

STEP 3. DATA AND THE churn_risk_flag TARGET (IMPORTANT, AVOID LEAKAGE)

The dataset has no churn label. Do NOT define the target from Recency and then use Recency as a feature, because the model would be trivially perfect and meaningless. Use a temporal split instead:

1. Create src\features.py with a function that builds the customer level table. Observation cutoff date default = 2024-10-01.
2. Features are computed ONLY from transactions with TransactionDate before the cutoff: Recency (days from last purchase to the cutoff), Frequency (number of unique TransactionID), Monetary (sum of TransactionValue), plus log1p versions of Frequency and Monetary, plus customer categorical attributes (Region, CustomerAgeGroup, CustomerGender as the most frequent value per customer, and preferred ProductCategory as the most frequent one).
3. Target: churn_risk_flag = 1 if the customer has no transaction from the cutoff date to the last date in the data (2024-12-30), otherwise 0. Exclude customers with no purchase before the cutoff.
4. Check the class balance. If the positive rate is outside roughly 8 percent to 45 percent, test other windows (for example cutoff 2024-11-01 for a 60 day window or 2024-09-01 for a 120 day window), choose the most workable one, and document the choice and the numbers. Use class_weight="balanced" where the model supports it and a stratified split.
5. Check missing values and duplicates explicitly and document the result (the dataset may have none; if so, say so, and still include a median imputer in the pipeline as the documented treatment).
6. Save the customer level modeling table to data\processed\churn_modeling_table.csv.

STEP 4. TRAINING SCRIPT AND SERIALIZED ARTIFACTS

Create src\train_models.py, runnable with: python src\train_models.py (from any folder). It must:

1. Split train and test 80/20, stratified, random_state=42.
2. Preprocessor (ColumnTransformer): StandardScaler on numeric features (Recency, Frequency_log, Monetary_log) and OneHotEncoder(handle_unknown="ignore") on the categorical features. Fit on train only.
3. Models: Logistic Regression, Decision Tree (small max_depth, as classical baseline), and ANN = MLPClassifier (for example hidden layers (32, 16), relu, adam, max_iter large enough to converge, random_state=42). Keep the hyperparameters simple and documented.
4. K-Means: reproduce the Milestone 2 model on full history RFM (Recency, log1p Frequency, log1p Monetary, own StandardScaler, k=3, random_state=42, n_init=10) with the same segment naming rule. Verify the result matches Milestone 2 (silhouette about 0.383, segment sizes 996, 729, 275). If it does not match, stop and tell me before continuing.
5. Serialize with joblib into models\: preprocessor.joblib, logistic_regression.joblib, decision_tree.joblib, ann_mlp.joblib, kmeans.joblib, kmeans_scaler.joblib, and kmeans_segment_map.joblib (cluster id to segment name). Also save models\metadata.json with: library versions, feature names and order, cutoff date, window, class balance, training date, random seed.
6. Evaluate on the TEST set only. Compute Accuracy, Precision, Recall, F1, ROC-AUC for the three classifiers; R-squared computed on predicted probabilities versus the true label, clearly labeled as such; Silhouette for K-Means only (use N/A where a metric does not apply). Optional: also score K-Means by treating the At Risk cluster as a churn prediction, clearly labeled. Save reports\benchmark_metrics.csv, plus confusion matrix counts, ROC curve points, and the ANN loss_curve_ (reports\ann_loss_curve.csv) so the app only reads files and does not retrain.
7. Print a short readable summary at the end.

STEP 5. THE STREAMLIT APP (app.py in the repo root)

General:
1. Page config wide layout. Dark theme from .streamlit\config.toml. Top KPI metric cards visible above the tabs: total customers, total transactions, total revenue, churn rate in the target, At Risk share from K-Means, best model by ROC-AUC with its value.
2. @st.cache_data for all CSV and metrics loading. @st.cache_resource for joblib model loading. st.session_state for the What-If state and the scenario history.
3. Models and scalers are loaded dynamically from models\ at runtime. If a file is missing or the scikit-learn version differs from metadata.json, show a clear st.error telling to run python src\train_models.py. No hardcoded predictions anywhere.
4. Keep UI code in app.py plus small helper modules in src\ (for example src\app_helpers.py) if app.py gets long. Tab labels: Data Explorer and EDA, Preprocessing and Correlation, Multi-Model Benchmark, What-If Inference.

Tab 1, Data Explorer and EDA: raw transactions sample (st.dataframe), df.describe() summary statistics, interactive Plotly histograms of customer level features with the churn_risk_flag as color overlay (selectbox to choose the feature), and a couple of categorical bar charts (for example churn rate by Region or ProductCategory).

Tab 2, Preprocessing and Correlation: show missing value check and treatment, StandardScaler effect (before versus after distributions or describe tables), log transform effect on Frequency and Monetary, One-Hot encoding of the categorical features (show the resulting feature matrix columns and a sample), and an interactive Pearson correlation heatmap in Plotly (include the target). Add one short plain English note explaining the temporal split and why it avoids leakage.

Tab 3, Multi-Model Benchmark: comparative table (Accuracy, Precision, Recall, F1, ROC-AUC, R-squared, Silhouette) for K-Means, Logistic Regression, Decision Tree, ANN, read from the saved metrics; side by side confusion matrices; ROC curves overlay; ANN loss convergence curve; and a short written interpretation under the loss curve and under the table (which model wins and why, whether the ANN overfits, how train versus test compare). Make it explicit that metrics are on the held out test set.

Tab 4, Real-Time What-If Inference Engine: sliders for Recency (days), Frequency (purchases) and Monetary (total spend), selectboxes for Region, Age group, Gender and Preferred category, a selectbox to choose the model (Logistic, Decision Tree, ANN) and a decision threshold slider (default 0.5). On every change: load the serialized preprocessor and chosen model, compute the churn probability, show it with st.progress and st.metric, also predict the K-Means segment with the K-Means scaler. Policy alerts: if probability is at or above the threshold use st.error with "re-engagement voucher" (suggest a stronger voucher when Monetary is high); if the probability is in a middle band use st.warning; otherwise st.success, and if the segment is Champions suggest the VIP voucher. Slider ranges must come from the data (min and max from the customer table). Add two preset buttons (typical Champion, typical At Risk) implemented with callbacks that set st.session_state BEFORE the widgets render (avoid the widget key conflict error). Keep a small scenario history table in st.session_state with a clear button.

STEP 6. TESTING (DO NOT SKIP)

1. Create tests\smoke_test.py that uses streamlit.testing.v1.AppTest to run the app, check there are no exceptions, change slider values, and confirm the prediction updates and the right alert type appears for a low risk and a high risk scenario. Run it.
2. Run the app headless: python -m streamlit run app.py --server.headless true, and check http://localhost:8501/_stcore/health returns ok. Then stop the server.
3. Test launching from a different working directory to confirm paths are correct.
4. Do a fresh clone test at the end: clone the pushed branch into a temp folder, create a new venv, install requirements.txt, and run the smoke test there. Report the result.
5. Confirm the models load only from .joblib files (no retraining at app start) and that first load time is reasonable (few seconds).

STEP 7. PULL REQUEST

1. Push feature/capstone-streamlit. Open a pull request to main with gh pr create (title: "feat: Streamlit capstone dashboard with serialized MLOps pipeline"). The description must be real and complete: objective, what changed per file, how to run (exact commands for Windows), how it was tested, model results summary, and a checklist. If gh is not installed, tell me and give me the GitHub compare link so I can open it manually: https://github.com/egallegosbc/Team3_UrbanMart_CapstoneProject/compare/main...feature/capstone-streamlit?expand=1
2. Tell me in Spanish to ask a teammate to review it. After their review and my explicit instruction, merge it into main (merge commit) and confirm main contains app.py, models, .streamlit, requirements.txt. Do not merge before that.

STEP 8. DOCUMENTS FOR ME (I WILL WRITE THE PDF REPORT AND THE SLIDES SEPARATELY IN ANOTHER TOOL)

Create these files in the repo folder docs\ and commit them in the docs commit:
1. docs\REPORT_FACTS.md with: final file tree; dataset facts; exact target definition, cutoff, window and class balance; the final metrics table (test set); confusion matrix counts; ROC-AUC values; ANN architecture, iterations and final loss; K-Means k, silhouette and segment sizes; the exact What-If rules and thresholds; three example scenarios with the real outputs (low risk, medium, high risk) that I can use in the live demo; exact commands to run training and the app; git branch name, git log --oneline of the work, PR URL and status; every problem you hit and how it was fixed.
2. docs\AI_LOG_DRAFT.md for the Responsible AI Pair Programming log: AI tool and model used (Claude Code), the verbatim prompt I gave you (this prompt), a summary of your key code suggestions and layout, a list of checks the human (me) should repeat and verify myself (scaler alignment, path handling, session_state keys, metrics on test set, no leakage), and the limitations or bugs found during testing (Streamlit reruns, widget key conflicts, path errors, version issues).
3. docs\DEFENSE_NOTES.md in Spanish, plain language: what each file does in two lines, the data flow from CSV to prediction, why the temporal split, why these models, how to demo the app in 8 minutes tab by tab, and 12 likely questions from peers or the professor with short answers.
4. Update README.md: project description, final structure, installation on Windows, how to train, how to run the app, results summary.

DEFINITION OF DONE

1. python -m streamlit run app.py opens all 4 tabs without errors on this laptop, from a clean venv.
2. Models load from .joblib at runtime and the What-If tab reacts live.
3. Benchmark shows test set metrics, confusion matrices, ROC curves and the ANN loss curve with interpretation.
4. Branch feature/capstone-streamlit pushed, conventional commits, PR open with a real description (merge only after my go ahead and a teammate review).
5. docs\ files created.

WHEN YOU FINISH, TELL ME IN SPANISH

1. What you built and the final metrics in a short table.
2. The exact command to start the app and the PR link.
3. Anything that needs my action (teammate review, collaborator access, screenshots of each tab for the report).
4. Any decision you made that I should know before I defend it.
```

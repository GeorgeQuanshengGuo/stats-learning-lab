# Stats Learning Lab

Live demo: [Open the Streamlit app](https://stats-learning-lab.streamlit.app/)

Please use synthetic or non-sensitive data only in the public demo.

A local Streamlit app I built for practicing applied statistics and basic machine learning workflows. This is a personal learning project based on my current understanding of data analysis: it organizes existing Python tools such as pandas, statsmodels, scikit-learn, Plotly, and Streamlit into a guided interface for uploading data, exploring variables, trying cleaning and transformations, fitting selected models, checking diagnostics, making simple predictions, and exporting a Markdown or HTML report. It is not a new statistical framework and it is intentionally not black-box AutoML; the user still chooses the target, features, model, preprocessing options, and interpretation path.

This first release should be understood as a learning-oriented starting point rather than a finished professional platform. As I continue studying and improving the project, I may add support for more topics, including time series and survival analysis.

Current release candidate:

```text
v1.0.0-rc1
```

## Key Features

- CSV and Excel upload with separate immutable `original_df` and editable `working_df`.
- Exploratory data analysis with schema detection, summary tables, missingness, charts, correlations, relationship explorer, target-aware EDA, and outlier checks.
- Confirm-before-apply cleaning for missing values.
- Variable transformations, target transformations, transformation suggestions, PCA, clustering labels, and anomaly flags without overwriting source columns.
- Statistical models with statsmodels: OLS linear regression, binary logistic regression, multinomial logistic regression, ordinal regression, Poisson regression, and Negative Binomial regression.
- Machine learning baselines with sklearn Pipelines: regression, binary classification, multiclass classification, cross-validation, tuning, feature importance, coefficients for applicable models, decision tree explanations, PDP/ICE, and calibration.
- Model diagnostics for overfitting, multicollinearity, learning curves, and OLS influence.
- Unified ModelRun records plus session-only fitted model artifacts for prediction.
- Prediction Console for saved models, including statistical intervals and optional ML bootstrap intervals for regression.
- Model Comparison tables grouped by task type.
- Markdown and HTML report export with logs, model summaries, formulas, predictions, diagnostics, limitations, and interpretation notes.
- Glossary and contextual help for common statistical and ML terms.
- Optional local workspace save/restore for trusted local sessions.

## Workflow

```text
Upload Data
  -> EDA
  -> Data Cleaning
  -> Transformations
  -> Statistical Models / Machine Learning
  -> Model Diagnostics
  -> Model Comparison
  -> Prediction
  -> Report Export
```

The workflow is flexible. You can revisit earlier steps whenever diagnostics or modeling results suggest that the working dataset needs more review.

## Installation

Use Python 3.10 or newer if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `python` is not available on your system, use `python3`.

## Run Locally

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser.

## Sample Datasets

Synthetic sample datasets are available in `sample_data/`.

Recommended starting files:

- `sample_data/regression_clean.csv`: clean regression workflow.
- `sample_data/regression_missing_outliers.csv`: missing values, outliers, and VIF checks.
- `sample_data/binary_classification.csv`: binary logistic regression and ML classification.
- `sample_data/multiclass_classification.csv`: multiclass classification.
- `sample_data/ordinal_example.csv`: ordinal regression.
- `sample_data/count_example.csv`: Poisson and Negative Binomial count models.
- `sample_data/edge_cases.csv`: messy column names, all-missing column, constant column, and ID-like fields.

See `docs/SAMPLE_DATASETS.md` for the purpose and expected use of each dataset.

## Validation Status

The project has an automated pytest suite covering core data handling, EDA helpers, cleaning, transformations, statistical models, ML workflows, prediction, reporting, and state integrity.

The latest validation run recorded:

```text
384 passed, 1 warning
```

The warning was a non-fatal joblib/loky CPU-core detection warning during clustering tests. For current local validation, run:

```bash
python3 -m pytest
```

Manual QA workflows are listed in `docs/E2E_QA_CHECKLIST.md`.

## Screenshots

Screenshots are not committed yet. Suggested placeholders:

- Home page workflow overview.
- Upload page after loading `regression_clean.csv`.
- EDA summary and chart card.
- Statistical model coefficient table and formula.
- Machine Learning results table.
- Prediction Console.
- Report preview.

## Known Limitations

- The app is designed for local trusted analysis, not multi-user production deployment.
- It does not replace study design, domain knowledge, or statistical judgment.
- Large datasets, high-cardinality categoricals, tuning, permutation importance, PDP/ICE, bootstrap intervals, and learning curves may be slow.
- Time series, survival models, mixed effects models, robust/clustered standard errors, advanced imputation, text modeling, PDF export, and Word export are not implemented.
- Feature importance, correlations, coefficients, and EDA relationships do not prove causality.
- Workspace snapshots use local pickle files and should only be restored from trusted local sources.
- Workspace save/restore is disabled by default for public deployments. For trusted local use only, set `STATS_LAB_ENABLE_WORKSPACE_SNAPSHOTS=true` before running the app.

See `docs/KNOWN_LIMITATIONS.md` for the full limitation list.

## Documentation

- `docs/QUICK_START.md`: 10-minute walkthrough.
- `docs/USER_GUIDE.md`: full user guide.
- `docs/TROUBLESHOOTING.md`: common issues and fixes.
- `docs/DEMO_SCRIPT.md`: presentation-style demo script.
- `docs/FEATURE_INVENTORY.md`: implemented feature inventory.
- `docs/STABILIZATION_PLAN.md`: validation-focused stabilization plan.

## Acknowledgements

I built this project with help from OpenAI Codex, especially for implementation support, debugging, documentation drafts, and test coverage. The project reflects my current learning process in statistics, data analysis, and machine learning.

## Disclaimer

This tool supports analysis practice and learning. It mainly brings together existing open-source Python libraries in one local workflow. It does not replace statistical judgment, domain expertise, or formal review. Results require interpretation, and causal conclusions require an appropriate study design beyond what the app can infer automatically.

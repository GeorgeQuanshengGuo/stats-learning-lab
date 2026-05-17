# v1.0 Release Notes

Release candidate: `v1.0.0-rc1`

Date: 2026-05-17

## Project Summary

Stats Learning Lab is a local Streamlit app I built for practicing applied statistics and basic machine learning workflows. It is a learning-focused project based on my current understanding of data analysis. Rather than inventing new statistical methods, it organizes existing pandas, statsmodels, scikit-learn, Plotly, and Streamlit functionality into a guided local interface.

The app is intentionally not black-box AutoML. Users choose the dataset, target variable, features, model family, preprocessing options, transformations, interpretation path, and report content.

This v1.0 release candidate is best understood as a starting point for learning and iteration. Future versions may add more topics as I keep studying, including possible support for time series and survival analysis.

## Major Features

### Data Upload And State Safety

- Upload CSV, XLSX, and XLS files.
- Store a protected `original_df`.
- Store an editable `working_df`.
- Apply cleaning and transformation changes only to `working_df` after user confirmation.
- Record cleaning and transformation logs.

### Exploratory Data Analysis

- Variable type detection.
- Summary tables.
- Missing value tables.
- Univariate charts.
- Correlation matrix and pairwise correlation table.
- Relationship explorer.
- Target-aware EDA recommendations.
- EDA-level outlier detection.

### Data Cleaning And Transformations

- Missing-value handling workflows.
- Log, log1p, square root, cube root, reciprocal, square, Box-Cox, and Yeo-Johnson transformations.
- Interaction terms, ratios, and polynomial features.
- Target transformation exploration.
- Transformation suggestions that are shown but not automatically applied.

### Statistical Models

- OLS Linear Regression.
- Binary Logistic Regression.
- Multinomial Logistic Regression.
- Ordinal Logistic Regression.
- Poisson Regression.
- Negative Binomial Regression.
- Coefficient tables, formulas, model statistics, metrics, diagnostics, and saved model artifacts where supported.

### Machine Learning

- ML regression baselines.
- ML binary classification baselines.
- ML multiclass classification baselines.
- Leakage-safe sklearn Pipelines and ColumnTransformers.
- Optional cross-validation.
- Optional GridSearchCV and RandomizedSearchCV tuning.
- Tree-based feature importance and optional permutation importance.
- Predictive coefficient tables for applicable sklearn linear/logistic models.
- Decision tree rules and tree explanations.
- PDP, ICE, and calibration helpers.
- PCA, clustering, and anomaly detection as exploratory modules.

### Model Management

- Unified lightweight ModelRun records.
- Session-only fitted model artifact registry.
- Model Comparison page with task-specific tables.
- Model Diagnostics page for overfitting, multicollinearity, learning curves, and OLS influence diagnostics.

### Prediction

- Prediction Console for saved fitted models.
- Readable saved model labels and metadata.
- Raw feature input fields.
- Regression point predictions.
- Statistical OLS mean confidence intervals and observation prediction intervals.
- Statistical Logit probability prediction where supported.
- Optional ML regression bootstrap empirical intervals.
- Classification predicted classes and probabilities where supported.

### Reporting And Help

- Markdown report export.
- HTML report export.
- Dataset overview, logs, model summaries, formulas, coefficient tables, diagnostics, prediction examples, and limitations where available.
- Searchable glossary.
- Contextual help and interpretation notes.
- User-facing documentation, quick start, troubleshooting guide, demo script, and manual QA checklist.

## Validation Performed

### Automated Tests

Command:

```bash
python3 -m pytest
```

Result:

```text
372 passed, 1 warning in 5.43s
```

The warning is from `joblib/loky` CPU-core detection during clustering tests. It does not indicate a test failure.

### Streamlit Smoke Test

Command:

```bash
streamlit run app.py --server.headless true --server.port 8521
```

Local check:

```bash
curl -I http://localhost:8521/
```

Result:

```text
HTTP/1.1 200 OK
```

In the current sandboxed environment, local port binding and local HTTP checks required escalated execution. This is a sandbox permission issue, not an app startup failure.

### Documentation Review

Confirmed present:

- `README.md`
- `docs/USER_GUIDE.md`
- `docs/QUICK_START.md`
- `docs/TROUBLESHOOTING.md`
- `docs/DEMO_SCRIPT.md`
- `docs/SAMPLE_DATASETS.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/E2E_QA_CHECKLIST.md`
- `docs/RELEASE_CHECKLIST.md`

### Release Hygiene Review

- `.gitignore` excludes generated outputs, reports, caches, local environment folders, local model artifacts, and workspace snapshots.
- No files larger than 1 MB were found outside ignored virtual/cache paths.
- Sample datasets are synthetic and documented.
- Local generated files such as caches and workspace snapshots are ignored and should not be included in a source release archive.

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the full list. The most important v1.0 limitations are:

- The app is for local trusted learning and analysis, not multi-user production deployment.
- The app reflects the author's current learning and should not be treated as a complete professional analytics platform.
- The app wraps and coordinates existing open-source Python libraries; it is not original statistical methodology.
- Results require interpretation and do not replace statistical judgment.
- Correlations, feature importance, and coefficients do not prove causality.
- Variable type detection is heuristic.
- Time series, survival models, mixed effects models, robust/clustered standard errors, advanced imputation, text modeling, and production ML monitoring are not implemented.
- ML classification probability uncertainty intervals are not implemented.
- PDF and Word export are not implemented.
- Workspace snapshots use pickle and should only be restored from trusted local files.
- Large datasets and runtime-heavy features can be slow.

## How To Run Locally

Use Python 3.10 or newer if possible.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL, usually:

```text
http://localhost:8501
```

## Sample Datasets Included

Documented sample datasets:

- `sample_data/regression_clean.csv`
- `sample_data/regression_missing_outliers.csv`
- `sample_data/binary_classification.csv`
- `sample_data/multiclass_classification.csv`
- `sample_data/ordinal_example.csv`
- `sample_data/count_example.csv`
- `sample_data/high_cardinality_example.csv`
- `sample_data/tiny_dataset.csv`
- `sample_data/edge_cases.csv`

Additional older synthetic sample files are also present and are not release blockers:

- `sample_data/regression_sample.csv`
- `sample_data/binary_classification_sample.csv`
- `sample_data/multiclass_sample.csv`
- `sample_data/eda_missing_values_sample.csv`

## Recommended Use Cases

- Learning applied statistics and machine learning workflows.
- Practicing EDA before modeling.
- Comparing statistical and ML model outputs on small to medium local datasets.
- Demonstrating why target selection, feature selection, preprocessing, diagnostics, and interpretation matter.
- Creating lightweight Markdown or HTML summaries of analysis sessions.

## Intentionally Not Included In v1.0

- Black-box AutoML.
- Automatic final model selection.
- Automatic causal conclusions.
- Automatic deletion of rows or outliers without user confirmation.
- Production deployment tooling.
- Authentication or multi-user permissions.
- Database/cloud data connectors.
- Time series forecasting.
- Survival analysis.
- Mixed effects models.
- Robust or clustered standard errors.
- Advanced imputation such as MICE or KNN imputation.
- XGBoost, LightGBM, CatBoost, SVM, neural networks, or deep learning.
- SHAP explanations.
- PDF or Word report export.
- Production model monitoring.

Some of these areas may be explored in future versions, especially time series and survival analysis, but they are intentionally outside the first release.

## Release Blockers

No release-blocking issues were found for `v1.0.0-rc1`.

## Recommended Tag

```text
v1.0.0-rc1
```

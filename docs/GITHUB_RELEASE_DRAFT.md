# GitHub Release Draft

## Release Title

```text
v1.0.0 - Semi-Automated Statistical and Machine Learning Analysis Workbench
```

## Suggested Git Tag

```text
v1.0.0
```

## Overview

Stats Learning Lab is a local Streamlit app for practicing applied statistics and basic machine learning workflows. It organizes common pandas, statsmodels, scikit-learn, Plotly, and Streamlit functionality into a guided interface for learning-oriented data analysis.

This release marks the first v1.0-ready version of the project. It supports data upload, exploratory analysis, cleaning, transformations, statistical modeling, machine learning baselines, diagnostics, prediction, model comparison, and Markdown/HTML report export.

The app is intentionally not black-box AutoML. Users choose the dataset, target, features, model family, preprocessing options, transformations, and interpretation path.

## Key Features

- CSV, XLSX, and XLS upload.
- Protected `original_df` plus editable `working_df`.
- EDA summaries, missing-value tables, charts, correlations, relationship explorer, target-aware EDA, and outlier checks.
- Confirm-before-apply missing-value cleaning.
- Feature transformations, target transformations, transformation suggestions, PCA, clustering, and anomaly detection.
- Statistical modeling with statsmodels:
  - OLS Linear Regression
  - Binary Logistic Regression
  - Multinomial Logistic Regression
  - Ordinal Logistic Regression
  - Poisson Regression
  - Negative Binomial Regression
- Machine learning workflows with sklearn Pipelines:
  - regression baselines
  - binary classification baselines
  - multiclass classification baselines
  - cross-validation
  - GridSearchCV / RandomizedSearchCV tuning
  - feature importance
  - predictive coefficient tables for applicable models
  - decision tree explanations
  - PDP, ICE, and calibration helpers
- Unified ModelRun records and session-only fitted model artifacts.
- Model Comparison page with task-specific metrics.
- Model Diagnostics page for overfitting, multicollinearity, learning curves, and OLS influence.
- Prediction Console for saved fitted models.
- OLS confidence intervals and prediction intervals where supported.
- Optional empirical bootstrap intervals for ML regression predictions.
- Markdown and HTML report export.
- Searchable glossary, contextual help, user guide, quick start, troubleshooting guide, and demo script.

## Validation Summary

Automated tests:

```bash
python3 -m pytest
```

Result:

```text
372 passed, 1 warning in 5.43s
```

The warning is from `joblib/loky` CPU-core detection during clustering tests. It does not indicate a test failure.

Streamlit smoke test:

```bash
streamlit run app.py --server.headless true --server.port 8521
curl -I http://localhost:8521/
```

Result:

```text
HTTP/1.1 200 OK
```

Release hygiene checks:

- `.gitignore` excludes generated outputs, reports, caches, model artifacts, workspace snapshots, and local environment files.
- Sample datasets are synthetic and documented.
- No large unnecessary files were found outside ignored virtual/cache paths.
- README, User Guide, Quick Start, Troubleshooting, Demo Script, known limitations, and release checklist are present.

## Installation And Run Instructions

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

## Sample Datasets

Included synthetic datasets:

- `sample_data/regression_clean.csv`
- `sample_data/regression_missing_outliers.csv`
- `sample_data/binary_classification.csv`
- `sample_data/multiclass_classification.csv`
- `sample_data/ordinal_example.csv`
- `sample_data/count_example.csv`
- `sample_data/high_cardinality_example.csv`
- `sample_data/tiny_dataset.csv`
- `sample_data/edge_cases.csv`

See `docs/SAMPLE_DATASETS.md` for each dataset's purpose, target variable, expected use, and intentionally included issues.

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the full list. Key limitations:

- Designed for local trusted learning and analysis, not multi-user production deployment.
- Does not replace statistical judgment, study design, or domain expertise.
- Correlations, coefficients, and feature importance do not prove causality.
- Variable type detection is heuristic.
- Large datasets and runtime-heavy features can be slow.
- Time series forecasting, survival analysis, mixed effects models, robust/clustered standard errors, advanced imputation, text modeling, and production ML monitoring are not included.
- XGBoost, LightGBM, CatBoost, SVM, neural networks, deep learning, and SHAP are not included.
- ML classification probability uncertainty intervals are not implemented.
- PDF and Word report export are not implemented.
- Workspace snapshots use local pickle files and should only be restored from trusted local sources.

## Recommended Next Steps

- Run the manual workflows in `docs/E2E_QA_CHECKLIST.md` with the included sample datasets.
- Create screenshots for README and GitHub release assets.
- Initialize the project as a git repository if it is not already one.
- Consider creating a lock file from a clean virtual environment for stricter reproducibility.
- Collect feedback from users learning statistics or applied ML and refine the UI copy where needed.
- Keep future changes focused on robustness, clarity, and correctness before adding new models.

## Disclaimer

This tool supports analysis and learning. It does not replace statistical judgment, domain expertise, or formal review. Results require interpretation, and causal conclusions require an appropriate study design beyond what the app can infer automatically.

## Suggested Git Commands

Review current state:

```bash
git status
```

Stage release files:

```bash
git add -A
```

Create release commit:

```bash
git commit -m "Prepare v1.0 release"
```

Create tag:

```bash
git tag v1.0.0
```

Do not run these commands until you have reviewed the changed files and confirmed this should become the GitHub v1.0 release.

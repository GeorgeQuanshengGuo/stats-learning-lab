# Stabilization Plan

Last updated: 2026-06-13

The main construction phase is complete. This plan intentionally avoids new major features. The next phase should prioritize correctness, robustness, reproducibility, performance awareness, and documentation.

## Current Baseline

Validated command:

```bash
python3 -m pytest
```

Result:

```text
384 passed, 1 warning in 6.42s
```

Known validation note:

- `python` may not exist in this environment. Use `python3`.
- The current warning is from joblib/loky CPU-core detection and does not indicate a failing test.

## Stabilization Goals

1. Confirm that `original_df` remains immutable across every page and workflow.
2. Confirm that `working_df` changes only after explicit user confirmation.
3. Confirm that ML preprocessing stays inside sklearn Pipelines and is fit only after train/test split or within CV folds.
4. Confirm that saved ModelRuns and model artifacts stay synchronized by `run_id`.
5. Confirm that user-facing outputs are honest about assumptions, uncertainty, and limitations.
6. Confirm that report export does not imply unsupported statistical conclusions.
7. Confirm that workspace save/restore works for local trusted snapshots and does not automatically restore without user confirmation.

## Validation Pass 1: Core State And Data Safety

Priority: high.

Steps:

1. Upload a small CSV and verify:
   - `original_df` and `working_df` are separate objects.
   - `uploaded_file_name` is stored.
   - schema/summary pages read `working_df`.
2. Apply one cleaning operation and verify:
   - `working_df` changes.
   - `original_df` remains unchanged.
   - `cleaning_log` has a readable entry.
3. Apply one transformation and verify:
   - a new column is created.
   - no source column is overwritten.
   - `transformation_log` has a readable entry.
4. Use reset working data and verify:
   - `working_df` returns to `original_df`.
   - cleaning/transformation logs are cleared as intended.

Recommended tests to keep running:

```bash
python3 -m pytest tests/test_cleaning.py tests/test_transformations.py tests/test_workspace_snapshot.py
```

## Validation Pass 2: EDA And Chart Stability

Priority: high.

Steps:

1. Test EDA with numeric, categorical, datetime, text, all-missing, and constant columns.
2. Confirm summary tables handle unsupported statistics with empty or `NaN` values rather than crashes.
3. Confirm univariate charts render for:
   - continuous numeric
   - discrete numeric
   - binary
   - nominal categorical
   - datetime
4. Confirm correlation matrix handles:
   - too few numeric variables
   - missing values
   - constant variables
5. Confirm relationship explorer handles:
   - numeric vs numeric
   - numeric vs categorical
   - categorical vs categorical
   - datetime vs numeric
6. Confirm chart cards do not produce duplicate Streamlit element IDs.

Recommended tests:

```bash
python3 -m pytest tests/test_summary.py tests/test_type_detector.py tests/test_correlation.py tests/test_relationships.py tests/test_outliers.py tests/test_chart_cards.py
```

## Validation Pass 3: Statistical Models

Priority: high.

Steps:

1. Fit one OLS model with numeric predictors.
2. Fit one OLS model with categorical predictors and confirm one-hot handling is stable.
3. Fit one binary Logit model and verify:
   - binary target validation
   - positive class mapping
   - threshold-dependent metrics and confusion matrix
4. Fit one multinomial model with three or more unordered classes.
5. Fit one ordinal model with confirmed category order.
6. Fit Poisson and Negative Binomial on a nonnegative integer-like target.
7. Confirm each saved run:
   - uses the unified ModelRun fields
   - creates an artifact when prediction is supported
   - appears in Model Comparison
   - appears in Prediction when artifact is usable

Recommended tests:

```bash
python3 -m pytest tests/test_linear_regression.py tests/test_logistic_regression.py tests/test_multinomial_logistic.py tests/test_ordinal_regression.py tests/test_count_regression.py
```

## Validation Pass 4: Machine Learning Models

Priority: high.

Steps:

1. Run one ML regression baseline and verify:
   - train/test split occurs before fitting.
   - full Pipeline contains preprocessing and estimator.
   - ModelRun and model artifact are saved.
2. Run one ML binary classifier and verify:
   - target has exactly two classes.
   - selected positive class maps correctly to encoded class 1.
   - split is stratified.
3. Run one ML multiclass classifier and verify:
   - target has three or more classes.
   - metrics are multiclass metrics and are not mixed with binary metrics.
4. Run 5-fold CV for regression and classification and verify CV metrics appear in Model Comparison.
5. Run one small tuning job and confirm the full Pipeline is passed into search.
6. Check optional runtime-heavy features on small data first:
   - permutation importance
   - PDP/ICE
   - calibration
   - learning curves
   - bootstrap intervals

Recommended tests:

```bash
python3 -m pytest tests/test_preprocessing.py tests/test_ml_regression.py tests/test_ml_classification.py tests/test_ml_multiclass_classification.py tests/test_ml_tuning.py
```

## Validation Pass 5: Unsupervised ML And Exploratory Modules

Priority: medium.

Steps:

1. Run PCA on numeric features and verify:
   - explained variance table
   - loadings table
   - component interpretation table
   - score table
   - optional PC score addition only after confirmation
2. Run KMeans, Agglomerative, and DBSCAN clustering and verify:
   - label table is clear
   - metrics handle edge cases
   - optional cluster label addition only after confirmation
3. Run anomaly detection and verify:
   - IQR and Z-score methods flag expected rows
   - Isolation Forest and LOF produce labels/scores
   - optional anomaly flag addition only after confirmation

Recommended tests:

```bash
python3 -m pytest tests/test_pca_module.py tests/test_clustering.py tests/test_anomaly_detection.py
```

## Validation Pass 6: Prediction, Intervals, And Artifacts

Priority: high.

Steps:

1. Fit one statistical OLS model, then use Prediction:
   - verify raw feature input fields
   - verify point prediction
   - verify mean CI and observation prediction interval
2. Fit one statistical Logit model, then use Prediction:
   - verify probability output
   - verify threshold behavior
   - verify probability CI if statsmodels exposes it
3. Fit one ML regression model, then use Prediction:
   - verify point prediction
   - optionally enable bootstrap empirical interval
   - confirm interval language is not called a classical CI
4. Fit one ML binary classifier, then use Prediction:
   - verify positive-class probability
   - verify selected threshold changes predicted class when applicable
5. Restore a saved workspace and confirm usable artifacts still appear when pickle restoration succeeds.

Recommended tests:

```bash
python3 -m pytest tests/test_prediction_service.py tests/test_statistical_intervals.py tests/test_ml_uncertainty.py tests/test_model_artifacts.py
```

## Validation Pass 7: Reports And Documentation

Priority: medium.

Steps:

1. Generate a report with no dataset and confirm friendly empty sections.
2. Generate a report after:
   - upload
   - one cleaning operation
   - one transformation
   - one model run
   - one prediction
3. Confirm Markdown and HTML downloads include:
   - dataset overview
   - logs
   - model runs
   - formulas where available
   - coefficient tables where available
   - prediction examples
   - limitations
4. Confirm report text does not make causal claims or unsupported statistical conclusions.

Recommended tests:

```bash
python3 -m pytest tests/test_report_builder.py tests/test_interpretation.py tests/test_formula_builder.py
```

## Manual Browser Smoke Test

Run the app and visit each major route:

```bash
streamlit run app.py
```

Routes to visit:

- `/`
- `/upload_data`
- `/eda`
- `/data_cleaning`
- `/transformations`
- `/statistical_models`
- `/model_comparison`
- `/machine_learning`
- `/report`
- `/prediction`
- `/model_diagnostics`
- `/glossary`

For each page, check:

- page loads without fatal error
- sidebar renders
- no duplicate Streamlit element ID errors
- empty state is friendly when no dataset exists
- controls keep state when switching pages where expected

## Release Readiness Checklist

Before calling the app stable:

- Full test suite passes.
- Manual route smoke test passes.
- One end-to-end regression workflow passes.
- One end-to-end binary classification workflow passes.
- One end-to-end statistical model plus prediction interval workflow passes.
- Workspace manual save/restore passes with a small trusted local dataset.
- Autosave prompt appears only when expected and never restores automatically.
- Known limitations are visible in docs and reports.

## Out Of Scope During Stabilization

Do not add these during the stabilization phase unless explicitly requested:

- new model families
- SHAP
- PDF/Word export
- cloud persistence
- multi-user authentication
- database-backed project storage
- automated model selection / AutoML
- causal inference modules
- advanced data cleaning workflows

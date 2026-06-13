# Machine Learning Leakage and Reproducibility Audit

Last updated: 2026-05-17

## Scope

This audit covers the current machine learning baseline workflows:

- reusable preprocessing in `src/modeling/preprocessing.py`
- ML regression in `src/modeling/machine_learning/regression.py`
- ML binary classification in `src/modeling/machine_learning/classification.py`
- ML hyperparameter tuning in `src/modeling/machine_learning/tuning.py`
- saved fitted artifacts in `src/core/model_artifacts.py`
- Prediction Console service in `src/prediction/prediction_service.py`
- empirical ML bootstrap uncertainty in `src/prediction/ml_uncertainty.py`

The audit focuses on data leakage, reproducibility, and whether model fitting modifies uploaded data.

## Summary Finding

The inspected ML workflows are structured correctly for the current baseline feature set:

- preprocessing is built as an unfitted sklearn `ColumnTransformer`
- model training uses a full sklearn `Pipeline`
- train/test split happens before model fitting
- cross-validation receives the full sklearn `Pipeline`
- hyperparameter tuning wraps `GridSearchCV` / `RandomizedSearchCV` around the full sklearn `Pipeline`
- binary classification uses stratified train/test splitting
- prediction uses saved fitted artifacts and does not silently refit regular predictions
- model fitting functions copy modeling data and do not mutate the caller's DataFrame

No analytics logic fixes were required during this audit. New tests were added to lock in these guarantees.

## Leakage Risks Checked

### 1. Preprocessing Pipeline Construction

Evidence:

- `build_preprocessing_pipeline(df, feature_columns, scale_numeric=False)` returns an unfitted `ColumnTransformer`.
- Numeric preprocessing uses median imputation and optional `StandardScaler`.
- Categorical preprocessing uses most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`.
- The helper does not call `fit`, `fit_transform`, or `transform`.

Added test coverage:

- `tests/test_ml_data_leakage.py::test_regression_preprocessing_is_fit_on_training_data_only`

This test uses a dataset where fitting on the full dataset would produce different median/scaler values and would learn a test-only category. The fitted pipeline learns the training-only median, training-only scaling mean, and does not learn the test-only category.

### 2. ML Regression Fitting

Evidence:

- `run_ml_regression_models()` creates a modeling copy with selected columns.
- `train_test_split()` is called before `x_train`, `x_test`, and pipeline fitting.
- `_build_regression_pipeline()` returns `Pipeline([("preprocessing", ColumnTransformer), ("model", estimator)])`.
- `pipeline.fit(x_train, y_train)` is called only after splitting.

Added test coverage:

- `tests/test_ml_data_leakage.py::test_regression_preprocessing_is_fit_on_training_data_only`
- `tests/test_reproducibility.py::test_ml_regression_results_are_reproducible_with_same_random_state`

### 3. ML Binary Classification Fitting

Evidence:

- `run_ml_binary_classification_models()` validates that the target has exactly two non-missing classes.
- The selected positive class is encoded as 1.
- `train_test_split(..., stratify=model_df[TARGET_BINARY_COLUMN])` is used.
- The fitted estimator is a full sklearn `Pipeline`.

Added test coverage:

- `tests/test_ml_data_leakage.py::test_ml_binary_classification_uses_stratified_split_and_pipeline`
- `tests/test_reproducibility.py::test_ml_binary_classification_results_are_reproducible_with_same_random_state`

### 4. Cross-validation

Evidence:

- Regression CV uses `KFold(..., shuffle=True, random_state=random_state)`.
- Binary classification CV uses `StratifiedKFold(..., shuffle=True, random_state=random_state)`.
- `cross_validate()` receives the full sklearn `Pipeline`, not a preprocessed matrix.
- Existing tests already monkeypatch `cross_validate()` to confirm the estimator is a `Pipeline`.

Relevant existing tests:

- `tests/test_ml_regression.py::test_regression_cross_validation_uses_full_pipeline`
- `tests/test_ml_classification.py::test_binary_classification_cross_validation_uses_full_pipeline`

### 5. Hyperparameter Tuning

Evidence:

- Tuning first creates train/test splits.
- Search objects are built with the full sklearn `Pipeline` as the estimator.
- `GridSearchCV` and `RandomizedSearchCV` fit only on `x_train`, `y_train`.
- CV splitters are `KFold` for regression and `StratifiedKFold` for binary classification.
- `best_estimator_` is the fitted full Pipeline saved for prediction.

Added test coverage:

- `tests/test_ml_data_leakage.py::test_hyperparameter_tuning_wraps_search_around_full_pipeline`
- `tests/test_reproducibility.py::test_randomized_tuning_is_reproducible_with_same_random_state`

Relevant existing tests:

- `tests/test_ml_tuning.py`

### 6. Prediction Console

Evidence:

- `predict_from_model_run()` requires a usable saved artifact.
- For sklearn ML models, prediction calls `artifact["fitted_pipeline"].predict(...)` or `predict_proba(...)`.
- The normal prediction path does not call `fit`.

Added test coverage:

- `tests/test_ml_data_leakage.py::test_prediction_service_uses_saved_pipeline_without_refitting`

Important distinction:

- ML bootstrap uncertainty in `src/prediction/ml_uncertainty.py` intentionally refits full Pipelines many times when the user explicitly enables empirical uncertainty intervals. This is not silent prediction refitting. Each bootstrap sample fits the full Pipeline so preprocessing stays inside the resampled workflow.

### 7. Data Immutability

Evidence:

- ML modeling helpers create modeling copies before dropping rows or encoding targets.
- The functions do not assign back into `original_df` or `working_df`.
- Page-level workflows call these helpers using `session_state["working_df"]`, but model fitting itself returns model outputs rather than a modified dataset.

Added and existing test coverage:

- `tests/test_ml_data_leakage.py::test_regression_preprocessing_is_fit_on_training_data_only`
- `tests/test_ml_data_leakage.py::test_ml_binary_classification_uses_stratified_split_and_pipeline`
- existing ML regression, classification, and tuning tests assert input DataFrames are unchanged.

## Reproducibility Checks

Added tests verify:

- Random Forest regression gives the same predictions and metrics with the same `random_state`.
- Random Forest binary classification gives the same predictions and probabilities with the same `random_state`.
- Randomized tuning gives the same selected parameters, CV score, and predictions with the same `random_state`.

Relevant file:

- `tests/test_reproducibility.py`

## Fixes Made

No ML implementation fixes were needed. The audit added regression tests and documentation only.

Files added:

- `tests/test_ml_data_leakage.py`
- `tests/test_reproducibility.py`
- `docs/ML_LEAKAGE_AUDIT.md`

## Validation Results

Command run:

```bash
python3 -m pytest
```

Result:

- 345 passed
- 1 warning from `joblib/loky` about physical CPU core detection in clustering tests
- no failures

## Remaining Limitations

- The audit tests representative ML paths rather than every possible page interaction.
- Exact reproducibility can still vary across different versions of numpy, pandas, scikit-learn, or joblib.
- Bootstrap uncertainty intentionally refits models; it is reproducible when `random_state` is fixed, but interval values depend on the bootstrap settings.
- Saved fitted model artifacts live in Streamlit session state. They are not designed as durable model files and are not exported into CSV or report downloads.
- Hyperparameter tuning is non-nested CV. The held-out test set is still preserved for final evaluation, but tuned performance estimates should be interpreted as baseline workflow diagnostics, not a final production validation protocol.
- Feature selection, if expanded in the future, must stay inside sklearn Pipelines or nested CV to avoid leakage.

# State Management Audit

Last updated: 2026-05-17

## Scope

This audit checks whether the current app protects uploaded data and keeps session state behavior predictable.

Reviewed areas:

- upload/session helpers in `src/core/state.py`
- upload page in `pages/01_upload_data.py`
- cleaning functions in `src/data/cleaning.py`
- transformation functions in `src/data/transformations.py`
- ML model fitting helpers
- prediction service in `src/prediction/prediction_service.py`
- report builder in `src/reporting/report_builder.py`
- ModelRun and model artifact registries

## Summary Finding

The current state-management design is sound:

- `original_df` is stored as a deep copy on upload.
- `working_df` is stored as a separate deep copy on upload.
- cleaning operations return new DataFrames and do not mutate inputs in place.
- transformation operations return new DataFrames and do not mutate inputs in place.
- confirmed cleaning/transformation actions replace `working_df` with a deep copy.
- `original_df` is not modified by cleaning, transformation, model fitting, prediction, or report generation.
- model fitting does not modify `working_df`.
- prediction and report generation are read-only with respect to `original_df` and `working_df`.
- `cleaning_log`, `transformation_log`, `model_runs`, and `model_artifacts` behave as expected.

No state-management bugs were found during this audit. New tests were added to protect these guarantees.

## Session State Keys Reviewed

| Key | Purpose | Audit result |
|---|---|---|
| `original_df` | untouched uploaded dataset | deep-copied on upload and preserved |
| `working_df` | dataset used by EDA, cleaning, transformations, modeling | separate deep copy; changed only by confirmed actions |
| `uploaded_file_name` | display metadata for current upload | set after successful upload |
| `cleaning_log` | human-readable cleaning operations | appended only when cleaning result is applied |
| `transformation_log` | transformation and feature engineering operations | appended only when transformation result is applied |
| `model_runs` | lightweight saved model metadata | appended after model fitting pages save runs |
| `model_artifacts` | session-only fitted models/pipelines keyed by `run_id` | linked to ModelRuns through matching `run_id` |
| `prediction_log` | saved prediction examples | read by report export |
| `pca_artifacts`, `clustering_artifacts`, `anomaly_artifacts` | exploratory ML artifacts | cleared when working data changes |

## Checks Performed

### 1. Upload Safety

Evidence:

- `set_uploaded_data(data)` stores `data.copy(deep=True)` into both `original_df` and `working_df`.
- The two copies are independent.
- Uploading a new dataset clears cleaning/transformation/prediction logs, exploratory artifacts, ModelRuns, and model artifacts.

Test:

- `tests/test_state_integrity.py::test_upload_stores_independent_original_and_working_copies`

### 2. Cleaning Does Not Mutate Inputs

Evidence:

- cleaning functions in `src/data/cleaning.py` return new DataFrames.
- fill operations use `df.copy(deep=True)`.
- drop operations use `.copy()` after row removal.
- `apply_cleaning_result(new_df, log_entry)` stores `new_df.copy(deep=True)` in `working_df`.

Tests:

- existing `tests/test_cleaning.py`
- `tests/test_state_integrity.py::test_cleaning_operation_and_state_apply_do_not_mutate_original_data`

### 3. Transformations Do Not Mutate Inputs

Evidence:

- transformations in `src/data/transformations.py` create a deep copy before adding new columns.
- transformations create new columns instead of overwriting source columns.
- `apply_transformation_result(new_df, log_entry)` stores `new_df.copy(deep=True)` in `working_df`.

Tests:

- existing `tests/test_transformations.py`
- `tests/test_state_integrity.py::test_transformation_operation_and_state_apply_do_not_mutate_original_data`

### 4. Model Fitting Does Not Modify Data

Evidence:

- ML modeling helpers create modeling copies internally.
- Statistical and ML model runs save metadata and fitted artifacts separately from the data.
- Model fitting returns outputs and saved artifacts, not modified datasets.

Tests:

- existing ML regression/classification/tuning tests assert input DataFrames are unchanged.
- `tests/test_state_integrity.py::test_model_fitting_records_run_and_artifact_without_mutating_session_data`

### 5. Prediction Is Read-only

Evidence:

- `predict_from_model_run()` builds a single-row input frame from supplied values.
- Regular prediction uses saved fitted artifacts.
- It does not write to `original_df` or `working_df`.

Test:

- `tests/test_state_integrity.py::test_prediction_does_not_mutate_original_or_working_data`

Note:

- Prediction pages may append to `prediction_log` after user action. That is expected state mutation, but it does not alter the data.

### 6. Report Generation Is Read-only

Evidence:

- `build_report_context()` builds dictionaries and record lists from the current session state.
- It computes summaries from the supplied DataFrame but does not assign back into the DataFrame.

Test:

- `tests/test_state_integrity.py::test_report_generation_does_not_mutate_original_or_working_data`

### 7. Logs And Model Registries

Evidence:

- `cleaning_log` is appended by `apply_cleaning_result()`.
- `transformation_log` is appended by `apply_transformation_result()`.
- ModelRuns are lightweight dictionaries created by `create_model_run()`.
- Fitted model objects live in `model_artifacts`, not inside exported ModelRun metadata.
- `model_artifacts` are keyed by the same `run_id` as the corresponding ModelRun.

Test:

- `tests/test_state_integrity.py::test_model_fitting_records_run_and_artifact_without_mutating_session_data`

### 8. Reset Working Dataset

Evidence:

- `reset_working_data()` restores `working_df` from `original_df.copy(deep=True)`.
- reset clears `cleaning_log` and `transformation_log`.
- reset clears latest displayed model/exploratory outputs and fitted `model_artifacts`.

Test:

- `tests/test_state_integrity.py::test_reset_working_dataset_restores_original_and_clears_edit_logs_and_artifacts`

Important behavior:

- Existing `model_runs` metadata can remain after working data changes, while fitted artifacts are cleared. This lets the app still display old saved results, but those runs cannot be used for prediction unless a matching artifact exists.

### 9. Old ModelRuns Without Artifacts

Evidence:

- `predict_from_model_run()` raises a clear `PredictionError` when no usable fitted artifact exists.
- Model Comparison can still display lightweight ModelRun metadata without a fitted object.

Test:

- `tests/test_state_integrity.py::test_old_model_run_without_artifact_is_handled_gracefully`

## Fixes Made

No code fixes were required. The audit added tests and documentation only.

Files added:

- `tests/test_state_integrity.py`
- `docs/STATE_MANAGEMENT_AUDIT.md`

## Validation Results

Targeted command:

```bash
python3 -m pytest tests/test_state_integrity.py
```

Result:

- 8 passed

Full validation command:

```bash
python3 -m pytest
```

Result:

- 353 passed
- 1 warning from `joblib/loky` about physical CPU core detection during clustering tests

## Remaining Limitations

- Streamlit session state is memory-backed for the current session. Browser refresh behavior depends on the app's workspace save/restore feature, not on native Streamlit persistence.
- Fitted `model_artifacts` are intentionally session-only and are not suitable as durable model files.
- Workspace snapshots use pickle for trusted local use only; they should not be treated as a secure import/export format for untrusted files.
- Old ModelRuns can remain after the underlying working data changes. This is useful for comparison history, but prediction requires a matching fitted artifact.
- Optional exploratory actions such as adding PCA scores, cluster labels, anomaly flags, or outlier flags intentionally mutate `working_df` only after user confirmation and should remain covered by manual UI validation.

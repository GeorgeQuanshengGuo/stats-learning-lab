# Sample Datasets

Last updated: 2026-05-17

These datasets are synthetic and intentionally small. They are designed for manual and automated validation of Stats Learning Lab. They do not contain real personal data.

## Overview

| File | Purpose | Target variable | Expected use | Intentional issues |
|---|---|---|---|---|
| `sample_data/regression_clean.csv` | Clean regression workflow | `target_score` | EDA, OLS, ML regression, model comparison, prediction, report export | Known near-linear relationship; categorical `region` predictor |
| `sample_data/regression_missing_outliers.csv` | Regression with cleaning, outliers, and VIF checks | `target_value` | Missing-value cleaning, outlier detection, VIF diagnostics, OLS/ML regression | Missing values, clear outliers, highly correlated predictors |
| `sample_data/binary_classification.csv` | Binary classification workflow | `purchased` | Binary Logit, ML binary classification, threshold checks, ROC/PR, prediction | Positive class is `yes`; mixed numeric/categorical predictors |
| `sample_data/multiclass_classification.csv` | Unordered multiclass workflow | `plan_type` | Multinomial Logit, ML multiclass classification, multiclass comparison | Three unordered classes: `Basic`, `Premium`, `Enterprise` |
| `sample_data/ordinal_example.csv` | Ordered target workflow | `satisfaction_level` | Ordinal regression, ordered-category EDA | Intended order: `low < medium < high` |
| `sample_data/count_example.csv` | Count regression workflow | `claims_count` | Poisson/Negative Binomial, overdispersion warning, expected count prediction | Nonnegative integer target with many zeros and high-count rows |
| `sample_data/high_cardinality_example.csv` | High-cardinality categorical stress case | `target_value` | Type detection, one-hot encoding behavior, ML preprocessing warnings | `customer_segment` has many unique categories |
| `sample_data/tiny_dataset.csv` | Graceful failure and warning checks | `target` | Small-sample warnings, split/CV edge cases | Only four rows |
| `sample_data/edge_cases.csv` | Type detection and EDA edge cases | `score value` | Type detection, summary tables, EDA robustness | All-missing column, constant column, ID-like columns, non-ASCII column names, spaces in names |

## Dataset Details

### `regression_clean.csv`

- Purpose: Validate the happy-path regression workflow.
- Target variable: `target_score`.
- Predictors: `sales`, `marketing_spend`, `training_hours`, `region`.
- Expected use:
  - Upload data.
  - Inspect EDA summaries and plots.
  - Fit OLS Linear Regression.
  - Fit ML regression baselines.
  - Compare model runs.
  - Use Prediction Console.
  - Export report.
- Known issues intentionally included:
  - None severe. This is a clean baseline dataset.
  - `region` is categorical so one-hot encoding should be exercised.

### `regression_missing_outliers.csv`

- Purpose: Validate cleaning, outlier detection, and diagnostics.
- Target variable: `target_value`.
- Predictors: `predictor_a`, `predictor_b`, `predictor_c`, `group`.
- Expected use:
  - Missing-value table.
  - Cleaning preview and confirmation.
  - EDA-level outlier detection.
  - OLS regression diagnostics.
  - VIF diagnostics because `predictor_a` and `predictor_b` are strongly correlated.
- Known issues intentionally included:
  - Missing target and predictor values.
  - One high outlier row.
  - One low leverage/outlier-style row.
  - Correlated predictors for multicollinearity checks.

### `binary_classification.csv`

- Purpose: Validate binary classification workflows.
- Target variable: `purchased`.
- Positive class: `yes`.
- Predictors: `age`, `income_score`, `visits`, `channel`.
- Expected use:
  - Binary target detection.
  - Statistical binary logistic regression.
  - ML binary classification.
  - Positive-class selection.
  - Threshold adjustment.
  - Confusion matrix, ROC curve, PR curve, calibration, prediction.
- Known issues intentionally included:
  - Small sample size, so metrics should be treated as demonstration only.
  - Target pattern is simple and may produce optimistic metrics.

### `multiclass_classification.csv`

- Purpose: Validate unordered multiclass classification.
- Target variable: `plan_type`.
- Classes: `Basic`, `Premium`, `Enterprise`.
- Predictors: `monthly_usage`, `support_tickets`, `team_size`, `industry`.
- Expected use:
  - Multiclass target detection.
  - Multinomial logistic regression.
  - ML multiclass classification.
  - Multiclass confusion matrix and class-wise metrics.
  - Prediction with class probabilities where available.
- Known issues intentionally included:
  - Classes are intentionally structured by usage/team size, so the task may be easy.
  - Dataset is still small.

### `ordinal_example.csv`

- Purpose: Validate ordered categorical target handling.
- Target variable: `satisfaction_level`.
- Intended order: `low < medium < high`.
- Predictors: `response_time_minutes`, `issue_count`, `agent_experience`, `service_channel`.
- Expected use:
  - Ordinal regression.
  - User-defined category order.
  - Ordered prediction probabilities.
  - Model comparison and report export.
- Known issues intentionally included:
  - The proportional odds assumption is not guaranteed.
  - Dataset is small and mostly illustrative.

### `count_example.csv`

- Purpose: Validate count regression.
- Target variable: `claims_count`.
- Predictors: `exposure`, `risk_score`, `vehicle_age`, `region`.
- Expected use:
  - Count-like target detection.
  - Poisson regression.
  - Negative Binomial regression.
  - Overdispersion warning.
  - Zero proportion warning.
  - Expected count prediction.
- Known issues intentionally included:
  - Many zero counts.
  - Several high counts to create possible overdispersion.

### `high_cardinality_example.csv`

- Purpose: Validate high-cardinality categorical behavior.
- Target variable: `target_value`.
- Predictors: `amount`, `visits`, `customer_segment`, `channel`.
- Expected use:
  - Type detection.
  - EDA categorical summaries.
  - ML preprocessing with one-hot encoding.
  - Report and limitation language around high-cardinality categories.
- Known issues intentionally included:
  - `customer_segment` has many levels relative to the sample size.
  - One-hot encoding can create many sparse features.

### `tiny_dataset.csv`

- Purpose: Validate graceful warnings and failures on very small data.
- Target variable: `target`.
- Predictors: `x1`, `x2`, `group`.
- Expected use:
  - Try EDA and basic summaries.
  - Try train/test split, CV, and models to confirm clear errors or warnings.
  - Confirm the app fails gracefully rather than crashing.
- Known issues intentionally included:
  - Only four rows.
  - `x1` and `x2` are perfectly correlated.
  - Too small for many modeling workflows.

### `edge_cases.csv`

- Purpose: Validate type detection and table robustness.
- Target variable: `score value` if a target is needed.
- Expected use:
  - Type detection.
  - Summary table.
  - Missing-value table.
  - Relationship explorer with awkward column names.
- Known issues intentionally included:
  - `all missing` is entirely missing.
  - `const column` is constant.
  - `Subject ID` and `id code` are ID-like.
  - `全名分组` uses non-ASCII characters.
  - Several columns contain spaces in their names.
  - `date text` mixes valid date strings, invalid text, and missing values.

## Suggested Manual Validation Flow

1. Start with `regression_clean.csv` for a full happy-path regression workflow.
2. Use `regression_missing_outliers.csv` to test cleaning, outlier detection, VIF, and OLS influence diagnostics.
3. Use `binary_classification.csv` to test positive-class logic and threshold behavior.
4. Use `multiclass_classification.csv`, `ordinal_example.csv`, and `count_example.csv` to test specialized statistical modules.
5. Use `high_cardinality_example.csv`, `tiny_dataset.csv`, and `edge_cases.csv` to test warnings, limitations, and graceful failures.

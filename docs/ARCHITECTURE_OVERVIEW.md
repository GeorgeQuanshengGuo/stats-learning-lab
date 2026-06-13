# Architecture Overview

## Concise File Tree

```text
.
├── AGENTS.md
├── README.md
├── app.py
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE_OVERVIEW.md
│   ├── CURRENT_PROGRESS.md
│   ├── NEXT_TASK_ML_BASELINE_SPEC.md
│   ├── PROJECT_HANDOFF.md
│   └── VALIDATION_STATUS.md
├── pages/
│   ├── 01_upload_data.py
│   ├── 02_analysis_plan.py
│   ├── 02_eda.py
│   ├── 03_data_cleaning.py
│   ├── 04_transformations.py
│   ├── 05_statistical_models.py
│   ├── 06_model_comparison.py
│   ├── 07_machine_learning.py
│   ├── 08_report.py
│   ├── 09_prediction.py
│   └── 10_model_diagnostics.py
├── sample_data/
│   ├── binary_classification_sample.csv
│   ├── eda_missing_values_sample.csv
│   ├── multiclass_sample.csv
│   └── regression_sample.csv
├── src/
│   ├── core/
│   │   ├── analysis_plan.py
│   │   ├── model_artifacts.py
│   │   ├── model_comparison.py
│   │   ├── model_run.py
│   │   ├── rigor.py
│   │   ├── workspace_snapshot.py
│   │   └── state.py
│   ├── data/
│   │   ├── cleaning.py
│   │   ├── loader.py
│   │   ├── transformation_suggestions.py
│   │   ├── transformations.py
│   │   └── type_detector.py
│   ├── eda/
│   │   ├── correlation.py
│   │   ├── missing.py
│   │   ├── outliers.py
│   │   ├── relationships.py
│   │   ├── summary.py
│   │   └── target_aware.py
│   ├── modeling/
│   │   ├── metrics.py
│   │   ├── preprocessing.py
│   │   ├── diagnostics/
│   │   │   ├── diagnostic_rules.py
│   │   │   ├── influence.py
│   │   │   ├── multicollinearity.py
│   │   │   └── overfitting.py
│   │   ├── interpretability/
│   │   │   ├── calibration.py
│   │   │   ├── coefficients.py
│   │   │   ├── pdp_ice.py
│   │   │   └── tree_explainer.py
│   │   ├── machine_learning/
│   │   │   ├── anomaly_detection.py
│   │   │   ├── classification.py
│   │   │   ├── clustering.py
│   │   │   ├── dimensionality_reduction.py
│   │   │   ├── feature_importance.py
│   │   │   ├── multiclass_classification.py
│   │   │   ├── regression.py
│   │   │   └── tuning.py
│   │   └── statistical/
│   │       ├── count_regression.py
│   │       ├── linear_regression.py
│   │       ├── logistic_regression.py
│   │       ├── multinomial_logistic.py
│   │       └── ordinal_regression.py
│   ├── prediction/
│   │   ├── input_builder.py
│   │   ├── ml_uncertainty.py
│   │   ├── prediction_service.py
│   │   └── statistical_intervals.py
│   ├── reporting/
│   │   ├── export_html.py
│   │   ├── export_markdown.py
│   │   ├── formula_builder.py
│   │   ├── interpretation.py
│   │   └── report_builder.py
│   └── visualization/
│       ├── correlation_plots.py
│       ├── diagnostic_plots.py
│       ├── eda_plots.py
│       ├── anomaly_plots.py
│       ├── clustering_plots.py
│       ├── influence_plots.py
│       ├── interpretability_plots.py
│       ├── model_plots.py
│       ├── outlier_plots.py
│       ├── pca_plots.py
│       ├── relationship_plots.py
│       ├── transformation_plots.py
│       └── tree_plots.py
└── tests/
    ├── test_calibration.py
    ├── test_count_regression.py
    ├── test_feature_importance.py
    ├── test_formula_builder.py
    ├── test_ml_classification.py
    ├── test_ml_coefficients.py
    ├── test_ml_multiclass_classification.py
    ├── test_ml_regression.py
    ├── test_ml_uncertainty.py
    ├── test_model_artifacts.py
    ├── test_model_comparison.py
    ├── test_multinomial_logistic.py
    ├── test_ordinal_regression.py
    ├── test_pdp_ice.py
    ├── test_prediction_service.py
    ├── test_statistical_intervals.py
    ├── test_tree_explainer.py
    └── ...
```

## Data Flow

```text
upload -> analysis plan -> EDA -> cleaning -> transformations -> modeling -> model comparison -> prediction -> diagnostics -> report export
```

1. Upload stores two deep copies of the uploaded dataset:
   - `original_df`: immutable source copy.
   - `working_df`: editable analysis copy.
2. Analysis Plan records the user's research question, goal, target/features, intended strategy, assumptions, and interpretation boundaries. It updates only analysis-plan session keys and never changes data.
3. EDA reads `working_df` and computes summaries, missingness, univariate plots, target-aware relationship summaries, correlation matrices, relationship explorer tables/plots, EDA-level outlier detection, and advisory data readiness checks.
4. Cleaning previews a new DataFrame, then updates `working_df` only after confirmation.
5. Transformations preview a new DataFrame with new columns, then update `working_df` only after confirmation.
6. Statistical modeling reads `working_df`, prepares train/test data, shows advisory model readiness checks, fits statsmodels models, saves lightweight ModelRuns, and saves fitted artifacts when prediction is supported.
7. Machine learning reads `working_df`, splits data before fitting, shows advisory model readiness checks, fits sklearn Pipelines, optionally computes CV and interpretability outputs, saves lightweight ModelRuns, and saves fitted Pipelines as artifacts.
8. PCA reads selected numeric features from `working_df` for exploratory dimension reduction. It does not change `working_df` unless the user confirms adding PC score columns.
9. Clustering reads selected numeric features from `working_df` for exploratory grouping. It does not change `working_df` unless the user confirms adding cluster labels.
10. Anomaly Detection reads selected numeric features from `working_df` for exploratory unusual-observation flags. It does not change `working_df` unless the user confirms adding an anomaly flag column.
11. Model Comparison reads `session_state["model_runs"]` and displays task-specific comparison tables, including validation warnings where available.
12. Prediction Console reads `session_state["model_runs"]` plus `session_state["model_artifacts"]` and predicts from saved fitted objects.
13. Model Diagnostics reads saved runs and artifacts for overfitting, multicollinearity, CV stability, learning curves, and OLS influence diagnostics where available.
14. Report Export reads the current session state and creates Markdown/HTML summaries, including the analysis plan, decision log, rigor checklist, data readiness summary, warnings, and reproducibility manifest.

## State Boundaries

- `original_df` is immutable after upload.
- `working_df` changes only through confirmed cleaning or transformation operations.
- `analysis_plan`, `analysis_decision_log`, `rigor_warnings`, and `analysis_stage_status` store advisory workflow metadata and do not change the data.
- Modeling and prediction must not mutate either DataFrame.
- ModelRun metadata lives in `session_state["model_runs"]`.
- Fitted model objects live only in `session_state["model_artifacts"]`.
- Prediction history lives in `session_state["prediction_log"]`.

## Modeling Layers

Statistical inference models live under `src/modeling/statistical/`:

- `linear_regression.py`: OLS.
- `logistic_regression.py`: binary Logit.
- `multinomial_logistic.py`: multinomial Logit.
- `ordinal_regression.py`: ordered Logit.
- `count_regression.py`: Poisson and Negative Binomial.

Predictive ML baselines live under `src/modeling/machine_learning/`:

- `regression.py`: regression baselines.
- `classification.py`: binary classification baselines.
- `multiclass_classification.py`: multiclass classification baselines.
- `feature_importance.py`: tree and permutation importance.
- `tuning.py`: GridSearchCV and RandomizedSearchCV tuning for selected regression and binary classification baselines.
- `dimensionality_reduction.py`: PCA for exploratory numeric dimension reduction.
- `clustering.py`: KMeans, Agglomerative Clustering, and DBSCAN for exploratory numeric clustering.
- `anomaly_detection.py`: IQR, Z-score, Isolation Forest, and Local Outlier Factor for exploratory anomaly detection.

ML preprocessing lives in `src/modeling/preprocessing.py`. It builds an unfitted sklearn `ColumnTransformer`; fitting must happen inside a full sklearn `Pipeline`.

Interpretability lives under `src/modeling/interpretability/` and `src/visualization/`:

- ML coefficient extraction for linear/logistic sklearn models.
- Decision tree rules and paths.
- PDP/ICE helpers.
- Binary classifier calibration helpers.
- Plotly and matplotlib visualization helpers.

Diagnostics live under `src/modeling/diagnostics/` and `src/visualization/`:

- `overfitting.py`: train/test and CV stability diagnostics.
- `multicollinearity.py`: VIF, predictor correlations, and condition numbers.
- `diagnostic_rules.py`: readable warning summaries.
- `influence.py`: OLS influence diagnostics.
- `diagnostic_plots.py` and `influence_plots.py`: visualizations for these checks.

Prediction lives under `src/prediction/`:

- raw input helpers
- prediction dispatch from saved artifacts
- statsmodels interval helpers
- ML bootstrap uncertainty helpers

## Helpers To Reuse

- `src/core/state.py`
  - Use `working_df` as the analysis input.
  - Never mutate `original_df`.
- `src/core/analysis_plan.py`
  - Use `create_analysis_plan()`, `update_analysis_plan()`, and `analysis_plan_is_recorded()` for plan state.
- `src/core/rigor.py`
  - Use `build_data_readiness()`, `build_model_readiness()`, `build_rigor_checklist()`, and `build_reproducibility_manifest()` for advisory workflow checks and report context.
- `src/core/model_run.py`
  - Use `create_model_run()` and `add_model_run_to_session()` for every saved model result.
- `src/core/model_artifacts.py`
  - Use `save_model_artifact()` for fitted objects needed later by prediction or interpretation.
- `src/core/model_comparison.py`
  - Add display support here when new task types or metrics are introduced.
- `src/modeling/diagnostics/`
  - Reuse existing diagnostics helpers when extending model health checks.
- `src/modeling/preprocessing.py`
  - Reuse `build_preprocessing_pipeline()` for sklearn ML models.
- `src/modeling/metrics.py`
  - Reuse existing regression, binary classification, and multiclass-style metric helpers where available.
- `src/reporting/formula_builder.py`
  - Reuse formula builders for statistical and interpretable ML models.
- `src/reporting/interpretation.py`
  - Keep explanation text rule-based and conservative.
- `src/prediction/prediction_service.py`
  - Extend this when adding prediction support for new artifact types.

## Leakage Boundary

All ML models must use scikit-learn `Pipeline` or `ColumnTransformer`.

Imputation, scaling, encoding, feature processing, and feature selection must be fit on training data only. Cross-validation must pass the full Pipeline to sklearn cross-validation utilities so preprocessing is fit independently inside each fold.

Do not fit preprocessing on the full `working_df` before train/test splitting or before cross-validation.

## Where To Add Future Features

Feature-stage development is currently paused. Future work should first verify current behavior, update tests, and only add new features when explicitly requested.

- Expanded tuning: extend `src/modeling/machine_learning/tuning.py` carefully. Keep the full Pipeline inside sklearn search objects and avoid black-box AutoML behavior.
- More statistical models: add separate modules under `src/modeling/statistical/` and extend `pages/05_statistical_models.py` plus Model Comparison and Prediction only where needed.
- More prediction intervals: extend `src/prediction/statistical_intervals.py` or `src/prediction/ml_uncertainty.py` and keep interval language honest.
- Advanced explainability: extend `src/modeling/interpretability/`; SHAP is intentionally not implemented yet.
- Report export: Markdown and HTML are implemented; PDF and Word are future features.

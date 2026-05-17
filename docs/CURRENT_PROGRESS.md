# Current Progress

Last updated: 2026-05-17

This is the quick-start status note for a fresh Codex session. Read this first, then use `PROJECT_HANDOFF.md` and `ARCHITECTURE_OVERVIEW.md` for deeper context.

## Current State

The app is a Streamlit-based learning tool for practicing applied statistics and machine learning workflows with:

- CSV/XLSX/XLS upload
- immutable `original_df`
- editable analysis copy in `working_df`
- EDA summaries, missingness tables, univariate plots, target-aware EDA, correlation matrix, relationship explorer, and EDA-level outlier detection
- missing-value cleaning with user confirmation and logs
- feature transformations, target transformations, and transformation suggestions
- statistical models with statsmodels:
  - OLS linear regression
  - binary logistic regression
  - multinomial logistic regression
  - ordinal logistic regression
  - Poisson regression
  - Negative Binomial regression
- sklearn machine learning baselines:
  - regression
  - binary classification
  - multiclass classification
- exploratory PCA / dimension reduction for numeric features
- exploratory clustering for numeric features with KMeans, Agglomerative Clustering, and DBSCAN
- exploratory anomaly detection with IQR, Z-score, Isolation Forest, and Local Outlier Factor
- optional ML cross-validation
- optional ML hyperparameter tuning with GridSearchCV or RandomizedSearchCV for selected regression and binary classification baselines
- ML feature importance, coefficient summaries for interpretable linear/logistic ML models, decision tree explanations, PDP/ICE, and calibration plots
- unified lightweight ModelRun storage
- session-only fitted model artifact registry
- Model Comparison page
- Prediction Console for saved fitted artifacts
- statistical prediction intervals for OLS and probability intervals where available for Logit
- empirical bootstrap uncertainty intervals for ML regression predictions
- rule-based model interpretation
- Markdown and HTML report export
- a Model Diagnostics page for overfitting, multicollinearity, learning-curve, and OLS influence diagnostics

Feature-stage development is currently paused. The next session should treat the repository as a stabilization/documentation handoff point unless the user explicitly resumes feature work.

## Streamlit Pages

- `pages/01_upload_data.py`: upload data and initialize `original_df` / `working_df`.
- `pages/02_eda.py`: dataset overview, summary tables, univariate EDA, missingness, and target-aware EDA.
- `pages/03_data_cleaning.py`: preview and confirm missing-value cleaning operations.
- `pages/04_transformations.py`: create new transformed variables and transformed targets.
- `pages/05_statistical_models.py`: fit statistical OLS, binary Logit, multinomial Logit, ordinal Logit, Poisson, and Negative Binomial models.
- `pages/06_model_comparison.py`: compare saved ModelRuns by task type and show concise rule-based interpretations.
- `pages/07_machine_learning.py`: run sklearn ML baselines, PCA, clustering, anomaly detection, CV, tuning, feature importance, coefficient summaries, tree explanations, PDP/ICE, and calibration inspection.
- `pages/08_report.py`: preview and download Markdown/HTML reports.
- `pages/09_prediction.py`: select saved fitted model artifacts and make manual predictions.
- `pages/10_model_diagnostics.py`: inspect saved model runs for overfitting risk, multicollinearity, cross-validation stability, learning curves, and OLS influence diagnostics where available.

## Important Invariants

- `original_df` must never be modified after upload.
- Cleaning and transformations update only `working_df`, and only after user confirmation.
- Modeling reads `working_df` but must not modify it.
- Fitted objects belong in `session_state["model_artifacts"]`, not in ModelRun exports.
- ModelRun entries must remain lightweight metadata for display, comparison, and reports.
- ML preprocessing must stay inside sklearn `Pipeline` / `ColumnTransformer`.
- Cross-validation must pass the full Pipeline so preprocessing is fit inside each fold.
- Prediction must construct a one-row DataFrame from raw feature names and must not modify `original_df` or `working_df`.
- Interpretability text and plots must not make causal claims.
- Report export is descriptive; no unsupported statistical interpretation should be invented.

## Core Modules

- `src/core/state.py`: session-state lifecycle and data immutability helpers.
- `src/core/model_run.py`: unified lightweight ModelRun creation and session storage.
- `src/core/model_artifacts.py`: session-only fitted model artifact registry.
- `src/core/model_comparison.py`: comparison tables for saved runs across regression, binary, multiclass, ordinal, and count tasks.
- `src/data/cleaning.py`: missing-value cleaning operations and logs.
- `src/data/transformations.py`: transformations, feature engineering, target transforms, and inverse transforms.
- `src/data/transformation_suggestions.py`: advisory transformation suggestions.
- `src/data/type_detector.py`: variable type detection.
- `src/eda/summary.py`, `src/eda/missing.py`: EDA summaries.
- `src/eda/correlation.py`: correlation matrices, pair tables, and high-correlation detection.
- `src/eda/target_aware.py`, `src/eda/relationships.py`: target-aware EDA and relationship summaries.
- `src/eda/outliers.py`: EDA-level univariate and multivariate outlier detection.
- `src/modeling/statistical/linear_regression.py`: statsmodels OLS.
- `src/modeling/statistical/logistic_regression.py`: statsmodels binary Logit.
- `src/modeling/statistical/multinomial_logistic.py`: statsmodels multinomial Logit.
- `src/modeling/statistical/ordinal_regression.py`: statsmodels ordered logistic regression.
- `src/modeling/statistical/count_regression.py`: Poisson and Negative Binomial count regression.
- `src/modeling/preprocessing.py`: unfitted sklearn preprocessing builder.
- `src/modeling/metrics.py`: reusable regression and classification metrics.
- `src/modeling/machine_learning/regression.py`: ML regression baselines.
- `src/modeling/machine_learning/classification.py`: ML binary classification baselines.
- `src/modeling/machine_learning/multiclass_classification.py`: ML multiclass classification baselines.
- `src/modeling/machine_learning/dimensionality_reduction.py`: PCA exploratory dimension reduction for numeric features.
- `src/modeling/machine_learning/clustering.py`: exploratory clustering for numeric features.
- `src/modeling/machine_learning/anomaly_detection.py`: exploratory anomaly detection for numeric features.
- `src/modeling/machine_learning/feature_importance.py`: tree and permutation importance helpers.
- `src/modeling/interpretability/coefficients.py`: coefficient tables for applicable sklearn linear/logistic models.
- `src/modeling/interpretability/tree_explainer.py`: decision tree rules, paths, and feature importance.
- `src/modeling/interpretability/pdp_ice.py`: PDP and ICE helpers.
- `src/modeling/interpretability/calibration.py`: binary classifier calibration helpers.
- `src/modeling/diagnostics/overfitting.py`: train/test gap and CV stability diagnostics.
- `src/modeling/diagnostics/multicollinearity.py`: VIF, predictor correlation, and condition-number diagnostics.
- `src/modeling/diagnostics/diagnostic_rules.py`: human-readable diagnostic rule summaries.
- `src/modeling/diagnostics/influence.py`: OLS residual, leverage, Cook's distance, DFFITS, and influence diagnostics.
- `src/prediction/input_builder.py`: raw feature input schema helpers.
- `src/prediction/prediction_service.py`: prediction from saved model artifacts.
- `src/prediction/statistical_intervals.py`: OLS and Logit interval helpers.
- `src/prediction/ml_uncertainty.py`: bootstrap empirical intervals for ML regression.
- `src/visualization/pca_plots.py`: PCA scree, cumulative variance, and PC1 vs PC2 plots.
- `src/visualization/clustering_plots.py`: clustering PCA scatter, cluster size, elbow, and silhouette plots.
- `src/visualization/anomaly_plots.py`: anomaly PCA scatter, score distribution, and feature boxplots.
- `src/visualization/correlation_plots.py`: correlation heatmap visualization.
- `src/visualization/outlier_plots.py`: EDA outlier plots.
- `src/visualization/diagnostic_plots.py`: overfitting, VIF, and learning-curve diagnostic plots.
- `src/visualization/influence_plots.py`: OLS influence diagnostic plots.
- `src/reporting/report_builder.py`: report context builder.
- `src/reporting/export_markdown.py`: Markdown report rendering.
- `src/reporting/export_html.py`: HTML report rendering.
- `src/reporting/interpretation.py`: rule-based model explanations.
- `src/reporting/formula_builder.py`: LaTeX formula builders.

## ModelRun And Artifacts

ModelRun entries are saved in `session_state["model_runs"]`. They are lightweight dictionaries with fields such as:

- `run_id`
- `timestamp`
- `task_type`
- `model_family`
- `model_name`
- `target`
- `features`
- `split_config`
- `preprocessing`
- `statistical_summary`
- `train_metrics`
- `test_metrics`
- `coefficient_table`
- `diagnostic_plot_keys`
- `feature_importance_table`
- `importance_type`
- `formula_latex`
- `notes`

Fitted model objects are saved separately in `session_state["model_artifacts"]`, keyed by the same `run_id`. Artifacts may include:

- fitted statsmodels result objects
- fitted sklearn Pipelines
- raw feature names
- transformed feature names
- target encoders or positive-class metadata
- coefficient tables
- formulas
- prediction and interval support metadata

Old ModelRuns without artifacts should still display in Model Comparison, but Prediction Console should treat them as unavailable for prediction.

## Validation Status

Latest validation command:

```bash
python3 -m pytest
```

Latest result:

```text
282 passed, 1 warning in 4.82s
```

The warning is a non-fatal joblib/loky CPU core detection warning from ML tests.

The local environment does not currently provide a `python` command, so use `python3` unless an alias is created.

## Known Limitations

- SHAP is not implemented.
- Zero-inflated count models are not implemented.
- Formal proportional-odds assumption tests are not implemented.
- Statistical multinomial prediction artifacts are limited compared with ML multiclass artifacts.
- Statistical modeling generally uses complete-case rows for selected variables.
- ML preprocessing choices are intentionally simple.
- Bootstrap intervals for ML regression are empirical and can be slow.
- ML classification probability uncertainty is not implemented.
- Reports export only Markdown and HTML; PDF and Word export are not implemented.
- Saved runs and fitted artifacts live only in Streamlit session state and are not persisted to disk.
- The folder is not currently a git repository.

## Recommended Next Task

Feature development is paused. A good next non-feature task is a careful manual UI walkthrough with the built-in sample datasets, then small bug fixes only. If feature development resumes later, keep fitted Python model objects session-only unless a safe serialization design is explicitly requested.

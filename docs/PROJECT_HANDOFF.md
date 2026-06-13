# Project Handoff

For a faster next-session briefing, read `docs/CURRENT_PROGRESS.md` first. This file is the detailed implementation handoff.

This repository is a Streamlit-based learning app for practicing applied statistics and machine learning workflows. The current app supports data upload, analysis planning, advisory rigor checklists, data/model readiness checks, EDA, target-aware EDA, correlation and relationship exploration, EDA-level outlier detection, missing-value cleaning, feature transformations, statistical modeling, machine learning baselines, model comparison, fitted artifact storage, prediction, model diagnostics, model interpretation, uncertainty displays, and Markdown/HTML report export.

Feature-stage development is currently paused. Future sessions should read this handoff, verify the current state, and avoid adding new feature scope unless the user explicitly resumes feature work.

## Current Pages

- `app.py`: Streamlit entry point. Initializes shared session state and shows a short landing message.
- `pages/01_upload_data.py`: Uploads CSV, XLSX, and XLS files. Uploaded data is stored as separate deep copies in `original_df` and `working_df`.
- `pages/02_analysis_plan.py`: Lets the user record the analysis title, research question, analysis goal, unit of analysis, target variable, candidate features, excluded columns, known ID/time/grouping columns, planned model family, validation strategy, missing-data strategy, transformation plan, assumptions to check, and interpretation boundaries. Changes append to `analysis_decision_log`.
- `pages/02_eda.py`: Shows dataset overview, per-column summary statistics, missing-value tables, basic univariate Plotly charts, target-aware EDA, correlation matrix, relationship explorer, and EDA-level outlier detection.
- `pages/03_data_cleaning.py`: Lets the user preview and confirm missing-value cleaning operations. Confirmed operations update `working_df` only and append to `cleaning_log`. Includes a reset button that restores `working_df` from `original_df`.
- `pages/04_transformations.py`: Lets the user preview and confirm feature transformations. Includes manual transformations, target transformation exploration, automatic transformation suggestions, and a transformation log. Confirmed transformations update `working_df` only and append to `transformation_log`.
- `pages/05_statistical_models.py`: Fits statsmodels OLS, binary Logit, multinomial Logit, ordinal Logit, Poisson, and Negative Binomial models. Saves lightweight ModelRuns and fitted artifacts where supported.
- `pages/06_model_comparison.py`: Reads saved ModelRun objects from session state, filters by task type, target, and model family, displays concise rule-based interpretations, displays task-specific comparison tables, supports clearing saved runs, and exports displayed tables as CSV.
- `pages/07_machine_learning.py`: Runs baseline sklearn regression, binary classification, and multiclass classification models on `working_df`. Also supports PCA / dimension reduction, clustering, and anomaly detection for selected numeric features, optional CV, optional GridSearchCV/RandomizedSearchCV tuning for selected regression and binary classification baselines, feature importance, interpretable ML coefficient tables, decision tree explanations, PDP/ICE, and calibration plots. Tuned estimators are saved as fitted artifacts and can be used by Prediction Console.
- `pages/08_report.py`: Builds a Markdown/HTML analysis report from the current session state, includes model interpretation text, previews the report, and provides Markdown and HTML download buttons.
- `pages/09_prediction.py`: Reads saved ModelRuns and fitted model artifacts, lets the user enter raw feature values, and returns predictions. Supports regression, binary classification, multiclass classification, ordinal classification, and count-regression style predictions when artifacts support them.
- `pages/10_model_diagnostics.py`: Reads saved ModelRuns and artifacts, then shows model metadata, overfitting diagnostics, CV stability summaries, multicollinearity diagnostics, optional learning curves for sklearn artifacts, and OLS influence diagnostics where available.

## Major Source Modules

### Core

- `src/core/state.py`: Initializes and updates Streamlit session state. Keeps `original_df` immutable by only writing confirmed cleaning and transformation results into `working_df`.
- `src/core/analysis_plan.py`: Creates and normalizes analysis plans, detects whether a meaningful plan exists, and builds timestamped decision-log entries when plan fields change.
- `src/core/rigor.py`: Provides advisory statistical rigor helpers: data readiness checks, model readiness checks, rigor checklist rows, and report reproducibility manifests.
- `src/core/model_run.py`: Defines the unified lightweight ModelRun dictionary structure and helper functions for creating, appending, reading, and clearing saved model runs.
- `src/core/model_artifacts.py`: Stores fitted model objects in `session_state["model_artifacts"]`, keyed by ModelRun `run_id`. This registry is session-only and is used by Prediction Console and saved-model interpretation.
- `src/core/model_comparison.py`: Builds overview, regression, binary classification, multiclass classification, ordinal classification, and count regression comparison tables from saved ModelRuns.

### Analysis Rigor

- `analysis_plan` is a lightweight session dictionary that records the user's intended analysis before modeling. It is advisory and does not block EDA or model fitting.
- `analysis_decision_log` records timestamped changes to plan fields, including old value, new value, and reason.
- `rigor_warnings` can hold advisory readiness or interpretation warnings generated by pages.
- `analysis_stage_status` is reserved for page-level workflow status.
- `build_data_readiness()` and `build_model_readiness()` in `src/core/rigor.py` are the reusable helpers that pages should call instead of duplicating warning logic.
- The report includes the analysis plan, decision log, rigor checklist, data readiness summary, warnings, and reproducibility manifest where available.

### Data And EDA

- `src/data/loader.py`: Loads uploaded CSV and Excel files into pandas DataFrames.
- `src/data/type_detector.py`: Classifies columns as `continuous_numeric`, `discrete_numeric`, `binary`, `nominal_categorical`, `ordinal_categorical_candidate`, `datetime`, `text`, `id_like`, or `constant`.
- `src/data/cleaning.py`: Implements missing-value cleaning helpers. Each function returns a new DataFrame and a human-readable log entry.
- `src/data/transformations.py`: Implements numeric transformations and feature engineering helpers, including log, log1p, square root, cube root, reciprocal, square, Box-Cox, Yeo-Johnson, interaction, ratio, polynomial features, and inverse transforms where available.
- `src/data/transformation_suggestions.py`: Builds advisory transformation suggestions for numeric variables. It never applies changes automatically.
- `src/eda/summary.py`: Builds dataset overview and one-row-per-variable summary tables.
- `src/eda/missing.py`: Builds missing-value summaries by variable and by row.
- `src/eda/correlation.py`: Builds numeric correlation matrices, pairwise correlation tables, and high-correlation pair summaries.
- `src/eda/relationships.py`: Builds target-aware and general relationship summary tables for numeric, categorical, and datetime combinations.
- `src/eda/target_aware.py`: Profiles selected target variables, classifies target type, and returns recommended analysis modules without fitting models.
- `src/eda/outliers.py`: Runs EDA-level IQR, Z-score, modified Z-score, Mahalanobis, Isolation Forest, and Local Outlier Factor outlier detection without deleting rows.

### Visualization

- `src/visualization/eda_plots.py`: Plotly charts for numeric, categorical, missing-value, and datetime EDA views.
- `src/visualization/correlation_plots.py`: Plotly heatmaps for correlation matrices.
- `src/visualization/relationship_plots.py`: Plotly charts for target-aware EDA relationships.
- `src/visualization/outlier_plots.py`: Boxplot, histogram, scatter, and PCA plots for EDA-level outlier detection.
- `src/visualization/transformation_plots.py`: Before/after histograms for transformation previews.
- `src/visualization/model_plots.py`: Model display plots, feature-importance bars, confusion matrices, and count observed-vs-predicted plots where applicable.
- `src/visualization/tree_plots.py`: Decision tree plotting helpers.
- `src/visualization/interpretability_plots.py`: PDP, ICE, calibration, and probability histogram plots.
- `src/visualization/diagnostic_plots.py`: Diagnostic plots for overfitting gaps, VIF, and learning curves.
- `src/visualization/influence_plots.py`: OLS influence plots for residuals, leverage, Cook's distance, and studentized residuals.
- `src/visualization/pca_plots.py`: PCA scree, cumulative variance, and PC1 vs PC2 scatter plots.
- `src/visualization/clustering_plots.py`: PCA-colored cluster scatter, cluster size, KMeans elbow, and silhouette plots.
- `src/visualization/anomaly_plots.py`: PCA-colored anomaly scatter, anomaly score distributions, and feature boxplots by anomaly status.

### Statistical Modeling

- `src/modeling/statistical/linear_regression.py`: Fits statsmodels OLS, supports categorical predictors through one-hot encoding, computes coefficient tables, R-style model statistics, train/test metrics, diagnostic plots, formulas, and optional original-scale metrics for transformed targets.
- `src/modeling/statistical/logistic_regression.py`: Fits statsmodels binary Logit, validates binary targets, maps the chosen positive class to 1, supports custom or F1-optimized thresholds, and computes inference tables, predictive metrics, confusion matrices, ROC/PR curves, and formulas.
- `src/modeling/statistical/multinomial_logistic.py`: Fits statsmodels multinomial logistic regression for unordered targets with more than two classes. Outputs class-specific coefficient tables, metrics, formulas, and saved ModelRuns.
- `src/modeling/statistical/ordinal_regression.py`: Fits statsmodels `OrderedModel` for ordered categorical targets. Outputs predictor coefficients, threshold/cutpoint parameters, odds ratios, metrics, proportional-odds warnings, formulas, and prediction metadata.
- `src/modeling/statistical/count_regression.py`: Fits Poisson and Negative Binomial count models for nonnegative integer-like targets. Outputs coefficient tables with incidence rate ratios, AIC/BIC, deviance/Pearson summaries, overdispersion checks, zero-inflation warnings, formulas, and expected-count prediction metadata.

### Machine Learning

- `src/modeling/preprocessing.py`: Builds an unfitted sklearn `ColumnTransformer`. Numeric features use median imputation and optional standard scaling. Categorical features use most-frequent imputation and one-hot encoding with unknown-category handling.
- `src/modeling/metrics.py`: Provides reusable regression, binary classification, and classification helper metrics that do not depend on Streamlit.
- `src/modeling/machine_learning/regression.py`: Fits selected sklearn regression baselines through full Pipelines. Supports optional KFold cross-validation and saves ModelRun-compatible results plus fitted artifacts.
- `src/modeling/machine_learning/classification.py`: Fits selected sklearn binary classification baselines through full Pipelines. Validates binary targets, maps the selected positive class to 1, uses stratified splitting, supports optional StratifiedKFold cross-validation, and saves artifacts.
- `src/modeling/machine_learning/multiclass_classification.py`: Fits selected sklearn multiclass classification baselines through full Pipelines. Computes macro/weighted metrics, log loss where feasible, confusion matrices, and saves prediction-capable artifacts.
- `src/modeling/machine_learning/feature_importance.py`: Extracts transformed feature names, tree-based importances, and optional permutation importances for fitted sklearn Pipelines.
- `src/modeling/machine_learning/tuning.py`: Tunes selected regression and binary classification baselines with GridSearchCV or RandomizedSearchCV. The search estimator is the full sklearn Pipeline, so preprocessing is fit inside each CV fold.
- `src/modeling/machine_learning/dimensionality_reduction.py`: Runs PCA as an unsupervised exploratory module for selected numeric features. It imputes missing numeric values with the median, scales by default, returns explained variance, loadings, and scores, and can add PC score columns only after user confirmation.
- `src/modeling/machine_learning/clustering.py`: Runs exploratory clustering for selected numeric features using KMeans, Agglomerative Clustering, or DBSCAN. It imputes numeric missing values with the median, scales by default, returns labels, cluster sizes, summaries, quality metrics, PCA projection scores, silhouette data, optional KMeans elbow data, and can add cluster labels only after user confirmation.
- `src/modeling/machine_learning/anomaly_detection.py`: Runs exploratory anomaly detection for selected numeric features using IQR rule, Z-score, Isolation Forest, or Local Outlier Factor. It returns anomaly labels, scores where available, summary counts, top anomalous rows, PCA projection data, feature boxplot data, and can add anomaly flags only after user confirmation.

### Interpretability

- `src/modeling/interpretability/coefficients.py`: Extracts coefficient tables for ML Linear Regression, Ridge, Lasso, and Logistic Regression. Does not add p-values or standard errors for sklearn models.
- `src/modeling/interpretability/tree_explainer.py`: Extracts decision tree text rules, feature importance, decision paths, and leaf summaries for tree regressors/classifiers.
- `src/modeling/interpretability/pdp_ice.py`: Computes PDP and ICE data for saved sklearn Pipelines.
- `src/modeling/interpretability/calibration.py`: Computes binary classifier calibration curves, probability histograms, and Brier scores where feasible.

### Diagnostics

- `src/modeling/diagnostics/overfitting.py`: Computes train/test gap and CV stability diagnostics for regression and classification ModelRuns.
- `src/modeling/diagnostics/multicollinearity.py`: Computes numeric VIF tables, predictor correlation tables, and condition numbers. VIF is labeled as a multicollinearity diagnostic, not an overfitting diagnostic.
- `src/modeling/diagnostics/diagnostic_rules.py`: Converts diagnostic rows into beginner-friendly warning summaries.
- `src/modeling/diagnostics/influence.py`: Computes OLS influence tables using statsmodels influence tools, including studentized residuals, leverage, Cook's distance, DFFITS, DFBETAS where available, and combined influence flags.

### Prediction

- `src/prediction/input_builder.py`: Builds raw input schemas from saved artifacts.
- `src/prediction/prediction_service.py`: Dispatches predictions from saved artifacts. It constructs single-row DataFrames from raw feature names and does not modify `original_df` or `working_df`.
- `src/prediction/statistical_intervals.py`: Provides OLS mean confidence intervals and observation prediction intervals, plus Logit probability confidence intervals where statsmodels supports them.
- `src/prediction/ml_uncertainty.py`: Provides optional bootstrap empirical intervals for ML regression predictions. These are labeled as empirical intervals, not classical confidence intervals.

### Reporting

- `src/reporting/report_builder.py`: Builds a report context from dataset overview, schema summary, missing summary, cleaning log, transformation log, saved model runs, comparison tables, and limitations.
- `src/reporting/export_markdown.py`: Renders a report context as Markdown.
- `src/reporting/export_html.py`: Renders a report context as standalone HTML.
- `src/reporting/interpretation.py`: Generates concise rule-based explanations for saved statistical and ML ModelRuns without external APIs or unsupported causal claims.
- `src/reporting/formula_builder.py`: Builds symbolic and estimated LaTeX formulas for linear, logistic, Poisson, sklearn linear/logistic, and decision tree summaries.

## Session State

- `original_df`: Deep copy of the uploaded dataset. This must never be modified by cleaning, transformation, modeling, prediction, or reporting operations.
- `working_df`: Deep copy used for analysis. Cleaning and transformation operations replace this value only after the user confirms a preview.
- `uploaded_file_name`: Upload-related state.
- `analysis_plan`: User-recorded analysis goal, target/features, strategy, assumptions, and interpretation boundaries.
- `analysis_decision_log`: Timestamped list of analysis-plan changes.
- `rigor_warnings`: Advisory workflow/readiness warnings collected for display/reporting.
- `analysis_stage_status`: Lightweight workflow status dictionary for future rigor UI.
- `schema`: Schema-like information may be computed on demand by `build_summary_table()` and `detect_variable_types()`. It may also be present if a page stores it.
- `cleaning_log`: List of confirmed cleaning log entries.
- `transformation_log`: List of confirmed transformation log entries, including target transformation metadata when applicable.
- `model_runs`: List of unified lightweight ModelRun dictionaries.
- `model_artifacts`: Dictionary of fitted model artifacts keyed by ModelRun `run_id`.
- `prediction_log`: List of manual predictions made through Prediction Console.
- `latest_linear_regression_result`: Temporary display result for the latest fitted OLS model.
- `latest_logistic_regression_result`: Temporary display result for the latest fitted binary Logit model.
- `latest_multinomial_logistic_result`: Temporary display result for the latest fitted multinomial Logit model.
- `latest_ordinal_logistic_result`: Temporary display result for the latest fitted ordinal model.
- `latest_count_regression_result`: Temporary display result for the latest fitted count model.
- `latest_ml_regression_results`: Temporary display results for the latest ML regression baselines.
- `latest_ml_classification_results`: Temporary display results for the latest ML binary classification baselines.
- `latest_ml_multiclass_results`: Temporary display results for the latest ML multiclass baselines.
- `latest_pca_result`: Temporary display result for the latest PCA analysis.
- `pca_artifacts`: Session-only summaries of recent PCA analyses.
- `latest_clustering_result`: Temporary display result for the latest clustering analysis.
- `clustering_artifacts`: Session-only summaries of recent clustering analyses.
- `latest_anomaly_result`: Temporary display result for the latest anomaly detection analysis.
- `anomaly_artifacts`: Session-only summaries of recent anomaly detection analyses.
- `latest_diagnostics_result`: May be used by diagnostics views for temporary display state.
- `cleaning_preview_df`, `cleaning_preview_log`: Temporary cleaning preview state.
- `transformation_preview_df`, `transformation_preview_log`, `transformation_preview_source`, `transformation_preview_new_column`, `transformation_preview_context`: Temporary transformation preview state.

## Data Immutability

`original_df` is created only when a file is uploaded through `set_uploaded_data()`. That function stores deep copies in both `original_df` and `working_df`.

Cleaning and transformation pages call state helpers that replace `working_df` with a deep copy of the confirmed preview result and append a log entry. They do not write to `original_df`.

Modeling and prediction pages read `working_df` or saved artifacts but do not replace either `original_df` or `working_df`.

The Analysis Plan page updates only `analysis_plan`, `analysis_decision_log`, `rigor_warnings`, and `analysis_stage_status`. It never mutates `original_df` or `working_df`.

The reset action in Data Cleaning restores `working_df` from a deep copy of `original_df` and clears cleaning and transformation logs.

## Transformation Workflow

The Transformations page has manual transformation, target transformation, transformation suggestion, and log sections.

All transformation helpers:

- return a new DataFrame and log entry
- never modify the input DataFrame in place
- create a new column instead of overwriting source columns
- validate mathematical eligibility
- record human-readable metadata in `transformation_log`

Target transformations store metadata such as:

- `original_target`
- `transformed_target`
- `method`
- `parameters`
- `inverse_transform_available`

Linear regression can detect transformed targets from this log and optionally compute original-scale test metrics when inverse transformation is available.

## Statistical Modeling Workflow

The Statistical Models page currently supports:

- Linear Regression: statsmodels OLS with categorical predictors one-hot encoded using training columns and test data reindexed to match.
- Binary Logistic Regression: statsmodels Logit with explicit positive-class mapping and threshold handling.
- Multinomial Logistic Regression: statsmodels MNLogit for unordered targets with more than two classes.
- Ordinal Logistic Regression: statsmodels OrderedModel for confirmed ordered targets.
- Count Regression: Poisson and Negative Binomial for nonnegative integer-like targets.

Statistical modules save lightweight ModelRuns and, where supported, fitted statsmodels artifacts. These artifacts are used later for formulas, explanations, and prediction intervals.

## Machine Learning Workflow

The Machine Learning page reads `working_df` and lets the user choose task type, target variable, feature variables, train/test split ratio, random state, numeric scaling, baseline models, optional cross-validation, and optional hyperparameter tuning for supported regression and binary classification baselines.

All ML models use sklearn `Pipeline` objects with preprocessing inside the Pipeline. Cross-validation receives the full Pipeline as the estimator, so imputation, scaling, and one-hot encoding are fit separately within each fold.

Implemented ML tasks:

- regression baselines
- binary classification baselines
- multiclass classification baselines
- PCA / dimension reduction for numeric features
- clustering for numeric features
- anomaly detection for numeric features

Interpretability available from the ML page:

- tree-based feature importance
- optional permutation importance
- coefficient tables and formulas for applicable sklearn linear/logistic models
- decision tree rules and plots
- PDP and ICE for saved model artifacts
- calibration plots for binary classifiers

PCA is exploratory and unsupervised. It does not save ModelRuns. It stores a temporary PCA result in session state and optionally saves PCA artifact summaries. PC scores are added to `working_df` only when the user confirms, and that action is recorded in `transformation_log`.

Clustering is exploratory and unsupervised. It does not save ModelRuns. It stores a temporary clustering result in session state and optionally saves clustering artifact summaries. Cluster labels are added to `working_df` only when the user confirms, and that action is recorded in `transformation_log`. The app should not overinterpret clusters as real-world categories.

Anomaly detection is exploratory and unsupervised. It does not remove rows automatically and does not save ModelRuns. It stores a temporary anomaly result in session state and optionally saves anomaly artifact summaries. Anomaly flags are added to `working_df` only when the user confirms, and that action is recorded in `transformation_log`. The app warns that flagged rows are unusual according to the selected method and features, not proven errors.

## Unified ModelRun Structure

Every saved model run should use `src/core/model_run.py`. Top-level fields are:

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

DataFrames are converted to list-of-records so saved runs are session-safe and easier to display or export.

## Model Artifact Registry

Fitted model objects are not stored inside ModelRun dictionaries. They live only in `session_state["model_artifacts"]`.

Artifacts may contain:

- `run_id`
- `model_name`
- `model_family`
- `task_type`
- `fitted_model`
- `fitted_pipeline`
- `target`
- `features`
- `transformed_feature_names`
- `preprocessing_summary`
- `target_encoder`
- `positive_class`
- `formula_latex`
- `coefficient_table`
- `prediction_supported`
- `interval_supported`
- `interval_method`
- `created_at`

Old ModelRuns without artifacts remain displayable in Model Comparison. Prediction Console should list only usable artifacts and warn when saved runs have no artifact.

## How Models Save Runs

Linear regression creates a ModelRun with `task_type="regression"` and `model_family="statistical"`, plus coefficient tables, diagnostics, statistical summaries, formulas, optional target transformation metadata, and a fitted statsmodels artifact.

Binary logistic regression creates a ModelRun with `task_type="binary_classification"` and `model_family="statistical"`, plus positive-class metadata, threshold metadata, coefficient and odds-ratio tables, formulas, metrics, diagnostics, and a fitted statsmodels artifact.

Multinomial logistic regression creates a ModelRun with `task_type="multiclass_classification"` and `model_family="statistical"`, plus class metadata, coefficient tables by class, metrics, formulas, and comparison-ready summaries.

Ordinal logistic regression creates a ModelRun with `task_type="ordinal_classification"` and `model_family="statistical"`, plus category order, predictor coefficients, threshold tables, metrics, formulas, and a prediction-capable artifact.

Count regression creates a ModelRun with `task_type="count_regression"` and `model_family="statistical"`, plus coefficient tables, IRRs, overdispersion/zero warnings, metrics, formulas, and expected-count prediction metadata.

ML regression creates one ModelRun per selected model with `task_type="regression"` and `model_family="machine_learning"`, plus train/test metrics, optional CV metrics, optional feature importance, optional ML coefficient tables/formulas, and a fitted Pipeline artifact.

ML binary classification creates one ModelRun per selected model with `task_type="binary_classification"` and `model_family="machine_learning"`, plus positive-class metadata, train/test metrics, optional CV metrics, optional feature importance, optional coefficient tables/formulas, and a fitted Pipeline artifact.

ML multiclass classification creates one ModelRun per selected model with `task_type="multiclass_classification"` and `model_family="machine_learning"`, plus target class metadata, macro/weighted metrics, log loss where feasible, confusion matrices, optional feature importance, and a fitted Pipeline artifact.

Tuned ML regression and binary classification runs use the same ModelRun structure with model names such as `Ridge (Tuned)` or `Logistic Regression (Tuned)`. Tuning metadata is stored under `preprocessing["tuning"]`, including search type, CV folds, scoring, best score, best params, runtime, and any overfitting warning. The best estimator Pipeline is saved in the artifact registry.

## Model Comparison Workflow

The Model Comparison page reads `get_model_runs()`, filters the list, and builds task-specific tables through `src/core/model_comparison.py`.

Currently supported comparison table groups:

- regression
- binary classification
- multiclass classification
- ordinal classification
- count regression

Regression, classification, ordinal, and count metrics are intentionally not merged into one universal score.

## Prediction Workflow

Prediction Console reads saved ModelRuns and model artifacts.

Supported prediction cases include:

- sklearn regression Pipelines
- sklearn binary classification Pipelines
- sklearn multiclass classification Pipelines
- statsmodels OLS artifacts with mean confidence intervals and observation prediction intervals
- statsmodels Logit artifacts with probability predictions and probability CIs where available
- ordinal regression artifacts with predicted category probabilities
- count regression artifacts with expected count predictions and expected-mean CIs where available

For ML regression, optional bootstrap empirical intervals can be computed. They are not classical confidence intervals and can be slow.

For ML classification, probability uncertainty is not implemented yet.

Each prediction log entry includes timestamp, run id, model name, input values, prediction, interval if available, and threshold if applicable.

## Report Export Workflow

The Report page reads the current Streamlit session state and builds an exportable report. It includes dataset overview, schema summary, missing value summary, cleaning log, transformation log, saved model runs, rule-based model interpretations, model comparison summaries, and limitations.

The report can be previewed in the app and downloaded as Markdown or HTML. PDF and Word export are not implemented.

## Diagnostics Workflow

The Model Diagnostics page reads saved ModelRuns and model artifacts. It does not modify `original_df` or `working_df`.

Implemented diagnostics include:

- regression train/test overfitting checks using RMSE and R-squared gaps
- classification train/test overfitting checks using accuracy, F1, and ROC AUC gaps where available
- CV stability summaries when CV metrics exist
- VIF and predictor correlation diagnostics for selected feature columns
- optional learning curves for sklearn artifacts, with a warning that this refits models
- OLS influence diagnostics for statsmodels OLS artifacts, including residual outliers, leverage, Cook's distance, and DFFITS-style influence checks

Influential observations are flagged for investigation only. They are not automatically wrong and should not be removed without domain justification.

## Known Limitations

- SHAP is not implemented.
- Random forest tree-by-tree explanations are not implemented.
- Zero-inflated count models are not implemented.
- Formal proportional-odds assumption tests are not implemented.
- Formal heteroskedasticity, autocorrelation, normality, and proportional-odds diagnostics remain limited.
- Statistical multinomial prediction support is limited compared with ML multiclass prediction support.
- Statistical modeling generally uses complete-case rows for selected variables instead of sklearn-style imputation pipelines.
- Target transformations are user-created columns. Their parameters are estimated before modeling, not inside a train-only preprocessing pipeline.
- ML bootstrap intervals are empirical and may be slow.
- ML classification probability uncertainty is not implemented.
- Model diagnostics are heuristic screening tools, not automatic proof of model validity or invalidity.
- Saved ModelRuns and artifacts live only in Streamlit session state and are not persisted to disk.
- Report export supports Markdown and HTML only; PDF and Word export are not implemented.

# Feature Inventory

Last updated: 2026-05-17

This document records the current implemented feature surface for Stats Learning Lab, a Streamlit learning app for practicing statistics and machine learning workflows. Status labels:

- `implemented`: Feature exists and has unit coverage or has been exercised during prior workflows.
- `partially implemented`: Feature exists, but has limited scope or incomplete end-to-end coverage.
- `needs validation`: Feature exists, but should be manually validated across realistic datasets before relying on it.
- `known issue`: A known limitation or risk requires explicit attention.

## Major Pages

| Page | Purpose | Status | Notes |
|---|---|---:|---|
| `app.py` | Home page, app initialization, shared sidebar, landing workflow, quick status | implemented | Home page has been simplified. Workspace save/restore controls live in the sidebar. |
| `pages/01_upload_data.py` | Upload CSV, XLSX, XLS files and create `original_df` plus `working_df` | implemented | `original_df` and `working_df` are stored as deep copies. |
| `pages/02_eda.py` | Univariate EDA, missing summaries, charts, correlation, relationship explorer, target-aware EDA, EDA outliers | implemented | Broad feature set. Complex UI paths still need manual smoke testing with varied datasets. |
| `pages/03_data_cleaning.py` | Preview and confirm missing-value cleaning operations | implemented | Updates `working_df` only after confirmation and writes `cleaning_log`. |
| `pages/04_transformations.py` | Manual transformations, target transformations, suggestions, logs | implemented | Adds new columns only after confirmation. No automatic transformation application. |
| `pages/05_statistical_models.py` | OLS, binary Logit, multinomial Logit, ordinal Logit, Poisson, Negative Binomial | implemented | Statsmodels-backed inference. Some advanced assumptions are warned about but not fully tested. |
| `pages/06_model_comparison.py` | Compare saved ModelRuns by task and model family | implemented | Handles regression, binary classification, multiclass, ordinal, and count runs separately. |
| `pages/07_machine_learning.py` | ML baselines, tuning, CV, PCA, clustering, anomaly detection, interpretability | implemented | Full sklearn Pipelines are used for ML models. PCA, clustering, and anomaly detection are exploratory and do not create ModelRuns. |
| `pages/08_report.py` | Markdown and HTML report preview/download | implemented | Includes available logs, model summaries, formulas, interpretations, predictions, and limitations. |
| `pages/09_prediction.py` | Prediction console from saved fitted artifacts | implemented | Shows readable model labels, metadata, formulas where available, and raw feature inputs. |
| `pages/10_model_diagnostics.py` | Overfitting, CV stability, multicollinearity, learning curves, OLS influence diagnostics | implemented | Learning curves may refit models and need manual runtime validation. |
| `pages/11_glossary.py` | Searchable glossary/help content | implemented | Terms are loaded from `assets/glossary_terms.json`. |

## Core And State

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/core/state.py` | implemented | Initializes session keys and protects `original_df` by only applying confirmed changes to `working_df`. |
| `src/core/model_run.py` | implemented | Unified lightweight ModelRun structure. |
| `src/core/model_artifacts.py` | implemented | Session-only fitted artifact registry keyed by ModelRun `run_id`. |
| `src/core/model_comparison.py` | implemented | Builds task-specific comparison tables. |
| `src/core/workspace_snapshot.py` | implemented | Local manual save, optional autosave, restore prompt, and start-fresh behavior. |
| Workspace snapshot persistence | partially implemented | Uses pickle for trusted local app snapshots. It is not a secure exchange format for untrusted files. |

## Upload

| Feature | Status | Notes |
|---|---:|---|
| CSV upload | implemented | Uses pandas loading helper. |
| Excel upload | implemented | Supports `.xlsx` and `.xls` when dependencies are installed. |
| Immutable original copy | implemented | `set_uploaded_data()` deep-copies upload into `original_df` and `working_df`. |
| Upload metadata | implemented | `uploaded_file_name` is stored in session state. |

## EDA

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/eda/summary.py` | implemented | Variable-level summary table with numeric and categorical statistics. |
| `src/eda/missing.py` | implemented | Missingness by variable and row. |
| `src/data/type_detector.py` | implemented | Detects continuous, discrete, binary, categorical, datetime, text, id-like, and constant columns. |
| `src/eda/correlation.py` | implemented | Pearson, Spearman, Kendall where feasible. |
| `src/eda/relationships.py` | implemented | Numeric-numeric, numeric-categorical, categorical-categorical, datetime-numeric summaries. |
| `src/eda/target_aware.py` | implemented | Target profiling and module recommendations without fitting models. |
| `src/eda/outliers.py` | implemented | IQR, Z-score, modified Z-score, Mahalanobis, Isolation Forest, LOF. |
| EDA charts | implemented | Plotly charts are used and many are wrapped in chart cards. |
| EDA manual breadth testing | needs validation | Large, sparse, high-cardinality, timezone-aware, and mixed-type datasets should be manually tested. |

## Cleaning

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/data/cleaning.py` | implemented | Drop rows, fill numeric mean/median, fill categorical mode/missing label. |
| Preview before commit | implemented | Page stores preview state before applying. |
| Cleaning log | implemented | Every confirmed cleaning action writes a human-readable log. |
| Original data safety | implemented | Cleaning only replaces `working_df`. |
| Advanced cleaning | partially implemented | Outlier deletion, duplicate handling, recoding, and type conversion workflows are not implemented as cleaning actions. |

## Transformations

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/data/transformations.py` | implemented | Log, log1p, sqrt, cube root, reciprocal, square, Box-Cox, Yeo-Johnson, interactions, ratios, polynomial features. |
| Transformation validation | implemented | Mathematical eligibility checks return clear errors. |
| Target transformation exploration | implemented | Supports continuous target previews and transformed target creation. |
| Inverse transform metadata | implemented | Available for supported target transformations. |
| `src/data/transformation_suggestions.py` | implemented | Suggests transformations without applying them. |
| Transformation log | implemented | Confirmed operations update `transformation_log`. |
| Automatic suggestions | partially implemented | Suggestions are heuristic and do not know the modeling target context. |

## Statistical Models

| Feature / Module | Status | Notes |
|---|---:|---|
| OLS linear regression | implemented | Statsmodels OLS, coefficient table, model stats, metrics, diagnostics, formulas, artifacts. |
| Binary logistic regression | implemented | Statsmodels Logit, positive-class mapping, threshold metrics, ROC/PR, formulas, artifacts. |
| Multinomial logistic regression | implemented | Statsmodels MNLogit for unordered multiclass targets. |
| Ordinal logistic regression | implemented | Statsmodels OrderedModel for ordered targets with user-confirmed order. |
| Poisson regression | implemented | Count model with IRRs, overdispersion checks, formula, prediction metadata. |
| Negative Binomial regression | implemented | Count model with dispersion-related output where available. |
| Statistical assumptions | needs validation | Assumption tests are partial. Several pages provide warnings rather than full formal tests. |
| Perfect separation/singular design handling | needs validation | Statsmodels edge cases should be manually tested on difficult datasets. |

## Machine Learning

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/modeling/preprocessing.py` | implemented | Builds unfitted ColumnTransformer with numeric/categorical preprocessing. |
| ML regression baselines | implemented | Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN. |
| ML binary classification baselines | implemented | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN. |
| ML multiclass classification baselines | implemented | Multinomial Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN. |
| Cross-validation | implemented | KFold / StratifiedKFold through full Pipelines. |
| Hyperparameter tuning | implemented | GridSearchCV / RandomizedSearchCV for selected regression and binary classification baselines. |
| Feature importance | implemented | Tree-based and optional permutation importance. |
| ML coefficients | implemented | Predictive coefficient tables for applicable sklearn linear/logistic models, without p-values. |
| Decision tree explanation | implemented | Rules, tree plot, feature importance, prediction paths where feasible. |
| PDP / ICE | implemented | Available for fitted sklearn Pipelines. |
| Calibration | implemented | Binary classifier calibration tables/plots where probabilities are available. |
| PCA | implemented | Exploratory numeric PCA with explained variance, loadings, scores, and optional PC score columns. |
| Clustering | implemented | KMeans, Agglomerative, DBSCAN with metrics and optional cluster label column. |
| Anomaly detection | implemented | IQR, Z-score, Isolation Forest, LOF with optional anomaly flag column. |
| Runtime safeguards | needs validation | Tuning, permutation importance, bootstrap, learning curves, PDP/ICE, and large datasets can be slow. |

## Diagnostics

| Feature / Module | Status | Notes |
|---|---:|---|
| Overfitting diagnostics | implemented | Train/test gap and CV stability summaries. |
| Multicollinearity diagnostics | implemented | VIF, predictor correlation table, condition number where feasible. |
| OLS influence diagnostics | implemented | Studentized residuals, leverage, Cook's distance, DFFITS, DFBETAS where available. |
| Learning curves | implemented | Optional and may refit models. |
| Formal assumption testing | partially implemented | Some assumptions are explained or warned about rather than fully tested. |

## Model Comparison

| Feature | Status | Notes |
|---|---:|---|
| Unified ModelRun display | implemented | Reads `session_state["model_runs"]`. |
| Task-specific tables | implemented | Avoids mixing regression and classification metrics. |
| Filters | implemented | Task type, target, model family. |
| CSV download | implemented | Exports currently displayed comparison tables. |
| Clear saved runs | implemented | Clears ModelRuns and artifacts. |

## Prediction

| Feature / Module | Status | Notes |
|---|---:|---|
| `src/prediction/input_builder.py` | implemented | Builds raw input widgets/specs from saved features. |
| `src/prediction/prediction_service.py` | implemented | Dispatches predictions from saved sklearn/statsmodels artifacts. |
| Statistical OLS intervals | implemented | Mean confidence intervals and observation prediction intervals via statsmodels. |
| Statistical Logit probability CI | partially implemented | Available when statsmodels prediction support provides needed frame columns. |
| ML regression bootstrap intervals | implemented | Optional empirical intervals, clearly labeled as non-classical. |
| ML classification uncertainty intervals | known issue | Not implemented. Probabilities are point estimates only. |
| Prediction log | implemented | Manual predictions append to `prediction_log`. |

## Report Export

| Feature / Module | Status | Notes |
|---|---:|---|
| Report context builder | implemented | Reads current session state. |
| Markdown export | implemented | Includes available sections and honest empty states. |
| HTML export | implemented | Standalone HTML output. |
| Formulas and coefficients | implemented | Included when saved in ModelRun/artifacts. |
| Prediction examples | implemented | Included from prediction log where available. |
| PDF / Word export | known issue | Not implemented. |
| Chart embedding in reports | partially implemented | Chart metadata can be collected, but full image embedding/export is limited. |

## Glossary And UI Help

| Feature / Module | Status | Notes |
|---|---:|---|
| `assets/glossary_terms.json` | implemented | Contains statistical, ML, diagnostics, prediction, and visualization terms. |
| `src/ui/glossary.py` | implemented | Loads, validates, searches glossary terms. |
| `src/ui/term_help.py` | implemented | Renders inline help and popovers. |
| `pages/11_glossary.py` | implemented | Searchable glossary page. |
| Global theme and CSS | implemented | Streamlit theme plus lightweight local CSS. |
| Reusable UI components | implemented | Page headers, cards, badges, empty states, chart cards. |
| Sidebar workspace controls | implemented | Manual save, optional autosave, restore prompt, start fresh. |

## Validation Status

Latest command run for this inventory:

```bash
python3 -m pytest
```

Result:

```text
372 passed, 1 warning in 5.43s
```

The warning is a joblib/loky CPU-core detection warning during clustering tests. It does not fail the suite.

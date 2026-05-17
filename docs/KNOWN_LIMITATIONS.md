# Known Limitations

Last updated: 2026-05-17

This document is intentionally honest. Stats Learning Lab is a learning and practice tool, not a replacement for statistical judgment, domain review, or production ML governance.

## General Scope

- The app is designed for local Streamlit workflows, not multi-user production deployment.
- The app does not implement black-box AutoML. Users must choose targets, features, models, and preprocessing options.
- Many outputs are exploratory. They should guide analysis, not become automatic conclusions.
- Large datasets may be slow or memory-heavy because many workflows use pandas, Plotly, statsmodels, and sklearn directly in the Streamlit process.
- The project directory is not currently a git repository, so repository-level diff/status tracking is not available unless initialized later.

## Data Type Limitations

- CSV and Excel uploads are supported. Database connections, cloud warehouses, Parquet, Feather, SAS, Stata, SPSS, JSON, and API sources are not implemented.
- Time series modeling is not implemented. Datetime columns are supported for EDA-style plots and summaries, not ARIMA/forecasting workflows.
- Text columns are detected, but NLP modeling and text feature extraction are not implemented.
- Image, audio, geospatial, graph/network, and nested data are not supported.
- Very high-cardinality categorical columns may create wide one-hot encoded design matrices and slow modeling.
- Mixed-type object columns can still require manual cleaning before reliable modeling.
- Missingness mechanisms are not diagnosed formally. The app summarizes missing values but does not determine MCAR/MAR/MNAR.
- Survey weights, clustered samples, panel data, repeated measures, and complex sampling designs are not supported.

## Upload And Workspace Persistence

- `original_df` is intended to remain immutable after upload. It is protected by app state conventions and tests, not by a database permission boundary.
- `working_df` changes only through confirmed cleaning/transformation actions, but all state lives inside the Streamlit process unless the user saves a workspace snapshot.
- Workspace snapshots use local pickle files for trusted local recovery. They are not a secure exchange format and should not be loaded from untrusted sources.
- Autosave is optional and local. Refreshing still asks before restoring; the app does not silently restore old state.
- Fitted model artifacts can be large and may not always pickle cleanly. Snapshot saving skips unpickleable values.

## EDA Limitations

- Variable type detection is heuristic and can be wrong, especially for coded categories, IDs, date-like strings, and numeric labels.
- Correlation does not imply causation.
- Pearson correlation may miss nonlinear relationships.
- Spearman and Kendall correlations are useful for monotonic relationships but still do not prove causality.
- Target-aware EDA recommends possible analysis modules, but it does not validate scientific appropriateness.
- Outlier detection flags unusual rows under selected methods. A flagged row is not automatically an error.
- Mahalanobis, Isolation Forest, and LOF behavior depends strongly on selected variables, scaling, and sample size.
- Plotly charts may become crowded with many categories or many points.

## Cleaning Limitations

- Implemented cleaning focuses on missing values.
- Duplicate-row handling is not implemented as a dedicated cleaning workflow.
- Type conversion, category recoding, string standardization, unit conversion, and date parsing workflows are not fully implemented as cleaning actions.
- Outlier deletion is intentionally not automated.
- Imputation methods are simple. Advanced imputation such as MICE, KNN imputation, or model-based imputation is not implemented.
- Cleaning choices are logged, but the app does not determine whether a cleaning choice is scientifically appropriate.

## Transformation Limitations

- Transformation suggestions are heuristic and target-agnostic.
- Transformations create new columns, but users still need to decide whether transformed variables are appropriate for modeling.
- Box-Cox requires strictly positive values.
- Yeo-Johnson can handle zero and negative values, but interpretability still requires care.
- Target transformations do not automatically make a model valid.
- Inverse-transformed metrics are only available when inverse transformation metadata is supported and applicable.
- PCA components are mathematical summaries of variance, not causal factors.

## Statistical Model Limitations

- Statsmodels models may fail on singular matrices, perfect separation, near-perfect separation, too many predictors, or insufficient data.
- OLS assumptions are not fully proven by the app. Diagnostic plots and warnings help users investigate, but do not replace formal review.
- Logistic regression can be unstable with rare events, separation, or very small class counts.
- Multinomial logistic regression assumes unordered classes and can be unstable with sparse class-predictor combinations.
- Ordinal regression depends on the proportional odds assumption. The app warns about this, but does not implement a full formal assumption test.
- Count models assume count-like nonnegative integer targets. Overdispersion and many zeros require careful interpretation.
- Zero-inflated Poisson and zero-inflated Negative Binomial models are not implemented.
- Robust standard errors, clustered standard errors, mixed effects models, generalized estimating equations, survival models, and time-series models are not implemented.
- Statistical coefficients are associations unless the study design supports causal claims.

## Machine Learning Limitations

- ML baselines are intentionally limited to common sklearn models.
- XGBoost, LightGBM, CatBoost, SVM, neural networks, and deep learning are not implemented.
- Hyperparameter tuning is basic GridSearchCV / RandomizedSearchCV, not nested CV and not AutoML.
- Cross-validation metrics can be unstable on small datasets or imbalanced classes.
- Feature importance does not imply causal importance.
- Permutation importance, PDP, and ICE can be misleading when predictors are highly correlated.
- Calibration is implemented for binary classifiers with probabilities, but probability calibration methods such as Platt scaling or isotonic calibration are not implemented as model training options.
- Class imbalance handling is limited. There is no dedicated resampling workflow such as SMOTE.
- ML models use train/test split and optional CV, but the app does not provide a production model monitoring pipeline.

## Diagnostics Limitations

- Overfitting rules are heuristic thresholds, not formal tests.
- VIF measures multicollinearity, not overfitting.
- VIF can become difficult to interpret after one-hot encoding or when many columns are generated.
- Learning curves refit models and may be slow.
- OLS influence diagnostics apply to OLS-style results, not every model family.
- Influential observations are not automatically wrong and should not be removed without domain justification.

## Prediction And Interval Limitations

- Prediction requires saved fitted artifacts in the current session or restored from a trusted local snapshot.
- Old ModelRun entries without artifacts can appear in comparison but cannot be used for prediction.
- OLS mean confidence intervals and observation prediction intervals rely on statsmodels assumptions.
- Prediction intervals are usually wider than mean confidence intervals and should not be treated as guarantees.
- Logistic probability confidence intervals are available only when statsmodels exposes the required prediction output.
- ML regression bootstrap intervals are empirical uncertainty intervals, not classical confidence intervals.
- ML classification probability uncertainty intervals are not implemented.
- Prediction Console expects raw feature values matching the saved artifact's required features. Missing or incompatible inputs can still fail.
- Predictions are model outputs, not decisions. The app does not encode operational policies or domain constraints.

## Report Export Limitations

- Markdown and HTML report export are implemented. PDF and Word export are not implemented.
- Reports summarize current session state; they are not a full reproducibility package unless the user also saves data, code, and workspace context.
- Reports include available formulas, coefficients, logs, predictions, and interpretations, but only for information saved in ModelRuns/artifacts.
- Chart image embedding is limited. Chart metadata can be captured, but full static chart export is not a complete reporting pipeline.
- Report interpretations are rule-based and conservative. They do not provide causal conclusions.
- Report text depends on what the user has run in the current session.

## UI And Browser Limitations

- Streamlit session state persists across page navigation but can reset on browser refresh unless a workspace snapshot is restored.
- Browser close/refresh warnings are not implemented because Streamlit does not provide a native robust unload-confirmation API.
- Some UI state is saved through Streamlit widget keys, but not every temporary interaction is guaranteed to survive refresh.
- Very wide tables and many charts may be visually crowded.
- Accessibility has been considered, but a formal accessibility audit has not been performed.

## Security And Trust Boundaries

- The app is intended for local trusted analysis.
- Uploaded data is handled inside the local Streamlit process.
- Workspace snapshots use pickle and must be treated as trusted local files only.
- The app should not execute user-provided JavaScript.
- The app does not implement authentication, authorization, encryption-at-rest, or audit logging.

## Current Validation Snapshot

Latest command:

```bash
python3 -m pytest
```

Result:

```text
372 passed, 1 warning in 5.43s
```

Remaining risk:

- The automated suite is broad, but manual end-to-end validation should still be repeated on realistic datasets before treating the app as stable.

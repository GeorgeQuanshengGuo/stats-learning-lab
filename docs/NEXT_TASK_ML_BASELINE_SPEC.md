# Machine Learning Baseline Status And Next Tasks

This file originally described the ML baseline implementation plan. The baseline is now implemented and has been extended with cross-validation, hyperparameter tuning, feature importance, interpretable ML coefficients, decision tree explanations, PDP/ICE, calibration, multiclass classification, fitted artifact storage, and Prediction Console integration.

For the fastest full-project briefing, read `docs/CURRENT_PROGRESS.md`.

## Implemented ML Pages And Modules

The main ML page is:

- `pages/07_machine_learning.py`

Reusable ML logic is implemented in:

- `src/modeling/preprocessing.py`
- `src/modeling/metrics.py`
- `src/modeling/machine_learning/regression.py`
- `src/modeling/machine_learning/classification.py`
- `src/modeling/machine_learning/multiclass_classification.py`
- `src/modeling/machine_learning/feature_importance.py`
- `src/modeling/machine_learning/tuning.py`
- `src/modeling/interpretability/coefficients.py`
- `src/modeling/interpretability/tree_explainer.py`
- `src/modeling/interpretability/pdp_ice.py`
- `src/modeling/interpretability/calibration.py`
- `src/visualization/model_plots.py`
- `src/visualization/tree_plots.py`
- `src/visualization/interpretability_plots.py`

Prediction and artifact integration is implemented in:

- `src/core/model_artifacts.py`
- `src/prediction/prediction_service.py`
- `src/prediction/input_builder.py`
- `src/prediction/ml_uncertainty.py`

## Supported ML Regression Baselines

- Linear Regression
- Ridge
- Lasso
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- KNN Regressor

Metrics:

- train/test RMSE
- train/test MAE
- train/test R-squared
- optional CV RMSE mean/std
- optional CV MAE mean/std
- optional CV R-squared mean/std

## Supported ML Binary Classification Baselines

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- KNN Classifier

Metrics:

- train/test accuracy
- train/test precision
- train/test recall
- train/test F1
- train/test ROC AUC when feasible
- train/test PR AUC when feasible
- optional CV accuracy mean/std
- optional CV precision mean/std
- optional CV recall mean/std
- optional CV F1 mean/std
- optional CV ROC AUC mean/std when feasible

## Supported ML Multiclass Classification Baselines

- Multinomial Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- KNN Classifier

Metrics:

- accuracy
- macro precision/recall/F1
- weighted precision/recall/F1
- log loss when probabilities are available
- confusion matrix
- class-wise metric tables

## Current Preprocessing Behavior

All ML models use sklearn Pipelines.

The shared preprocessing helper builds an unfitted `ColumnTransformer`:

- numeric columns: median imputation, optional `StandardScaler`
- categorical columns: most-frequent imputation, `OneHotEncoder(handle_unknown="ignore")`

The preprocessing object is fit only inside the model Pipeline after train/test split. Cross-validation also receives the full Pipeline, so preprocessing is fit inside each fold.

## ModelRun And Artifact Integration

Every ML model result is saved as a unified ModelRun:

- `task_type="regression"`, `task_type="binary_classification"`, or `task_type="multiclass_classification"`
- `model_family="machine_learning"`
- model name, target, features, split config, preprocessing metadata, train metrics, and test metrics
- optional cross-validation metrics stored in `test_metrics`
- optional coefficient table, formula, feature importance, and interpretation notes

Fitted sklearn Pipelines are saved separately in `session_state["model_artifacts"]` under the same `run_id`. These artifacts power Prediction Console and saved-model interpretation. Do not place fitted objects into reports or CSV exports.

Saved ML runs appear in `pages/06_model_comparison.py` alongside statistical runs. Regression, binary classification, multiclass classification, ordinal classification, and count regression tables remain separate.

## Interpretability Status

Implemented:

- tree-based feature importance
- optional permutation importance on test data
- coefficient tables for ML Linear Regression, Ridge, Lasso, and Logistic Regression
- LaTeX estimated formulas for applicable ML linear/logistic models
- decision tree rules, plots, feature importance, and one-row decision paths
- PDP and ICE for saved sklearn Pipelines
- calibration curves and probability histograms for binary classifiers

Not implemented:

- SHAP
- random forest tree-by-tree explanations
- causal interpretation

## Hyperparameter Tuning Status

Basic user-controlled hyperparameter tuning is implemented for selected regression and binary classification baselines.

Implemented scope:

- Use sklearn `GridSearchCV` or `RandomizedSearchCV`.
- Pass the full Pipeline to the search object.
- Use KFold for regression and StratifiedKFold for classification.
- Fit the search only on training data after the train/test split.
- Report best parameters, CV score, train metrics, and test metrics.
- Save tuned results through the existing ModelRun structure.
- Save fitted best estimators in the model artifact registry.
- Make tuned runs appear in Model Comparison, Prediction Console when prediction is supported, and Report Export.

## Current Pause

Feature-stage development is currently paused. Future sessions should not treat this file as an instruction to implement more ML features unless the user explicitly resumes feature work.

## Next Recommended Task

When feature development resumes, the next major ML feature could be a cautious model persistence/export design or richer tuning controls for the already-supported models. Keep fitted Python objects session-only unless serialization is explicitly designed and tested.

Out of scope until explicitly requested:

- XGBoost
- LightGBM
- SHAP
- neural networks
- SVM
- automated model selection across all possible settings

## Guardrails For Future ML Work

- Do not mutate `original_df`.
- Do not mutate `working_df` during model fitting or prediction.
- Do not fit preprocessing outside sklearn Pipeline.
- Do not fit learned preprocessing on the full dataset before splitting.
- Do not turn the app into black-box AutoML; the user should choose target, features, models, and options.
- Add or update tests for every new modeling helper.

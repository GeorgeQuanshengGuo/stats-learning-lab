# User Guide

This guide explains how to use Stats Learning Lab from upload to report export. The app is meant for learning and practice, not as a replacement for statistical judgment.

## 1. Upload Data

Open **Upload Data** and choose a CSV or Excel file.

When a file is uploaded, the app creates two copies:

- `original_df`: the protected original upload. This should not be modified.
- `working_df`: the active analysis copy. Cleaning and transformations update this only after confirmation.

Use the preview table to confirm that rows, columns, and data types look reasonable before moving on.

Recommended next step: open **Exploratory Data Analysis**.

## 2. Read EDA

Open **Exploratory Data Analysis** to understand the dataset before modeling.

Typical checks:

- Review the summary table for variable type, missing values, unique counts, numeric summaries, and top categories.
- Review missing-value tables to decide whether cleaning is needed.
- Inspect histograms, boxplots, and categorical bar charts.
- Use the correlation matrix for numeric relationships.
- Use the relationship explorer for pairwise comparisons.
- Use target-aware EDA when you already have a target variable in mind.
- Use outlier tools to flag unusual rows without deleting anything automatically.

Important cautions:

- EDA is descriptive.
- Correlation does not imply causation.
- Outlier flags do not prove that a row is wrong.

Recommended next step: clean missing values or create transformations if needed.

## 3. Clean Data

Open **Data Cleaning** to handle missing values in the working dataset.

Available cleaning actions include:

- drop rows missing a selected column
- drop rows with any missing value
- fill numeric missing values with mean or median
- fill categorical missing values with mode
- fill categorical missing values with a label such as `Missing`

The page previews the result before applying it. The original uploaded dataset remains unchanged. Each confirmed action is recorded in `cleaning_log`.

Recommended next step: return to EDA to confirm that the cleaning result looks sensible.

## 4. Transform Variables

Open **Transformations** to create new variables without overwriting source columns.

Common transformations:

- log, log1p, square root, cube root, reciprocal, square
- Box-Cox and Yeo-Johnson
- interaction terms
- ratio features
- polynomial features
- target transformations for continuous outcomes

The app checks whether each transformation is mathematically allowed. For example, log requires positive values and Box-Cox requires strictly positive values.

Transformation suggestions are available, but they are not applied automatically. Confirmed transformations update `working_df` and write to `transformation_log`.

Recommended next step: fit a statistical or ML model using the transformed variable if appropriate.

## 5. Fit Statistical Models

Open **Statistical Models** when you need interpretable statistical inference such as p-values, confidence intervals, AIC, BIC, or coefficient tables.

Available statistical models:

- OLS Linear Regression for continuous targets
- Binary Logistic Regression for binary targets
- Multinomial Logistic Regression for unordered multiclass targets
- Ordinal Logistic Regression for ordered categorical targets
- Poisson Regression for count targets
- Negative Binomial Regression for count targets with possible overdispersion

The typical workflow is:

1. Select a target variable.
2. Select feature variables.
3. Choose train/test split settings.
4. Fit the model.
5. Review coefficients, formulas, metrics, diagnostics, and warnings.

Statistical coefficients are associations unless your study design supports causal claims.

Recommended next step: check diagnostics, compare saved runs, and use prediction if needed.

## 6. Fit Machine Learning Models

Open **Machine Learning** when the goal is predictive performance rather than statistical inference.

Available ML workflows include:

- regression baselines
- binary classification baselines
- multiclass classification baselines
- cross-validation
- GridSearchCV / RandomizedSearchCV tuning
- feature importance
- predictive coefficient tables for applicable linear/logistic sklearn models
- decision tree rules
- PDP, ICE, and calibration tools
- PCA, clustering, and anomaly detection as exploratory modules

ML models use sklearn Pipelines so imputation, scaling, encoding, cross-validation, and tuning avoid fitting preprocessing on the full dataset before splitting.

Important cautions:

- ML coefficients do not include p-values or classical confidence intervals.
- Feature importance does not prove causality.
- Tuned models can still overfit.

Recommended next step: open **Model Diagnostics** and **Model Comparison**.

## 7. Use Diagnostics

Open **Model Diagnostics** to inspect saved model runs.

Diagnostics can include:

- train/test gap and overfitting warnings
- cross-validation stability
- VIF and predictor correlation for multicollinearity
- learning curves for sklearn models
- OLS influence diagnostics such as leverage and Cook's distance

Diagnostics are warning tools, not automatic decisions. For example, a high Cook's distance row should be investigated, not deleted automatically.

Recommended next step: revise features, transformations, or model choice if diagnostics show problems.

## 8. Compare Models

Open **Model Comparison** to compare saved ModelRuns.

The page keeps regression, binary classification, multiclass, ordinal, and count metrics separate so incompatible metrics are not mixed.

Use filters for:

- task type
- model family
- target variable

You can download the displayed comparison table as CSV.

## 9. Use Prediction

Open **Prediction** after fitting at least one model.

The Prediction Console:

- lists saved model runs that have fitted artifacts
- shows readable model metadata
- shows formulas when available
- asks for raw feature values
- returns predictions from the saved fitted model

For regression models, prediction may include:

- point prediction
- OLS mean confidence interval
- OLS observation prediction interval
- optional ML bootstrap empirical interval

For classification models, prediction may include:

- predicted class
- positive-class probability
- class probabilities
- selected threshold

Predictions do not modify `original_df` or `working_df`.

## 10. Export Reports

Open **Report** to preview and download a Markdown or HTML report.

Reports can include:

- dataset overview
- working dataset status
- schema and missing-value summaries
- cleaning log
- transformation log
- model summaries
- formulas and coefficient tables
- train/test and CV metrics
- diagnostics summary
- prediction examples and intervals
- model comparison tables
- limitations and interpretation notes

Reports summarize the current session. They do not export fitted model objects or replace statistical review.

## 11. Workspace Save And Restore

The sidebar includes local workspace save/restore controls.

Default behavior after refresh is a clean state. If a saved workspace exists, the app can ask whether to restore it. Manual save is always available, and autosave is optional.

Only restore workspace snapshots from trusted local files.

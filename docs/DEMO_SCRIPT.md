# Demo Script

Use this script to present the project to someone else. The demo is designed for 10-20 minutes.

## Demo Goal

Show that Stats Learning Lab is a learning-focused analysis app, not black-box AutoML. The user stays in control of data cleaning, transformations, target selection, features, model choice, diagnostics, prediction, and report export.

## Setup

Start the app:

```bash
streamlit run app.py
```

Use these sample datasets:

- `sample_data/regression_clean.csv`
- `sample_data/binary_classification.csv`
- optionally `sample_data/regression_missing_outliers.csv`

## Opening

Suggested narration:

> This is a small learning app for practicing applied statistics and machine learning workflows. It helps with common analysis steps, but it does not choose conclusions for the user. The original uploaded dataset stays protected, and all cleaning or transformation changes happen only to a working copy after confirmation.

Show the Home page briefly:

- Point out the workflow.
- Point out the sidebar dataset status.
- Mention local workspace save/restore.

## Part 1: Upload And EDA

1. Go to **Upload Data**.
2. Upload `regression_clean.csv`.
3. Show the preview.
4. Go to **Exploratory Data Analysis**.
5. Show:
   - summary table
   - missing value table
   - histogram or boxplot
   - correlation matrix
   - relationship explorer

Suggested narration:

> Before modeling, the app encourages users to inspect types, missingness, distributions, and relationships. This is descriptive. It does not make causal claims from correlations or plots.

## Part 2: Statistical Model

1. Go to **Statistical Models**.
2. Choose Linear Regression.
3. Select `target_score` as the target.
4. Select predictors such as `sales`, `marketing_spend`, `training_hours`, and `region`.
5. Fit the model.
6. Show:
   - coefficient table
   - p-values and confidence intervals
   - R-squared and AIC/BIC
   - LaTeX formula
   - diagnostic plots

Suggested narration:

> Statistical models use statsmodels when inference outputs are needed. The coefficient table supports interpretation, but coefficients are associations unless the study design supports causality.

## Part 3: Machine Learning Baseline

1. Go to **Machine Learning**.
2. Choose Regression.
3. Use the same target and predictors.
4. Select several models, for example:
   - Linear Regression
   - Random Forest Regressor
   - KNN Regressor
5. Run the models.
6. Show:
   - train/test metrics
   - cross-validation option if enabled
   - feature importance for tree-based models
   - coefficient table for applicable linear models

Suggested narration:

> ML models are built with sklearn Pipelines. Preprocessing is fitted after the train/test split or inside each cross-validation fold, which helps avoid data leakage.

## Part 4: Diagnostics And Comparison

1. Go to **Model Diagnostics**.
2. Select a saved run.
3. Show:
   - overfitting diagnostics
   - VIF/multicollinearity for applicable models
   - influence diagnostics for OLS if available
4. Go to **Model Comparison**.
5. Show task-specific comparison tables.

Suggested narration:

> The app separates regression and classification metrics. It avoids ranking every model with one universal score because different tasks need different metrics.

## Part 5: Prediction

1. Go to **Prediction**.
2. Select a saved model run.
3. Show the readable model label and metadata.
4. Enter one row of feature values.
5. Predict.

Suggested narration:

> Prediction uses the saved fitted artifact from the session. For OLS, the app can show mean confidence intervals and observation prediction intervals. For ML regression, empirical bootstrap intervals can be enabled, but they are not classical confidence intervals.

## Part 6: Report Export

1. Go to **Report**.
2. Show the preview.
3. Download Markdown.
4. Download HTML.

Suggested narration:

> The report summarizes what has actually been run: dataset overview, logs, model summaries, formulas, metrics, predictions, diagnostics, and limitations. It is honest when a section has no data.

## Optional Classification Demo

If time allows:

1. Upload `binary_classification.csv`.
2. Fit statistical binary logistic regression.
3. Choose positive class `yes`.
4. Show odds ratios, threshold behavior, ROC/PR charts, and confusion matrix.
5. Fit an ML binary classifier.
6. Use Prediction to show positive-class probability.

Suggested narration:

> For classification, the positive class and threshold matter. The app makes those choices visible instead of hiding them.

## Closing

Suggested summary:

> The app automates repetitive analysis mechanics, but it keeps important choices visible. It supports learning, reproducibility, and transparent interpretation. It does not replace statistical judgment or domain review.

## Demo Checklist

- [ ] Upload succeeds.
- [ ] EDA renders without errors.
- [ ] Statistical model fits and saves a run.
- [ ] ML model fits and saves a run.
- [ ] Model Comparison shows saved runs.
- [ ] Prediction works for a saved artifact.
- [ ] Report preview and downloads work.
- [ ] Limitations and cautions are visible.

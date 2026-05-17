# Quick Start: 10-Minute Walkthrough

This walkthrough uses `sample_data/regression_clean.csv`.

## Before You Start

Install dependencies and run the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL, usually `http://localhost:8501`.

## Minute 1: Upload Data

1. Open **Upload Data**.
2. Upload `sample_data/regression_clean.csv`.
3. Confirm that the dataset preview appears.

Expected result:

- Rows and columns are shown.
- The app creates an immutable original copy and a working copy.

## Minute 2: Inspect EDA

1. Open **Exploratory Data Analysis**.
2. Review the summary table.
3. Review the missing-value table.

Expected result:

- Numeric variables have mean, standard deviation, quantiles, skewness, and kurtosis.
- Categorical variables show top value and frequency.

## Minute 3: View Charts

1. Select a numeric variable such as `target_score`.
2. Inspect the histogram and boxplot.
3. Select a categorical variable such as `region`.
4. Inspect the bar chart.

Expected result:

- Charts render in chart cards with context and cautions.

## Minute 4: Check Relationships

1. Open the correlation section.
2. Select numeric variables.
3. Use Pearson or Spearman correlation.
4. Open Relationship Explorer and compare `sales` with `target_score`.

Expected result:

- A heatmap and pairwise correlation table appear.
- A scatter plot appears for numeric-vs-numeric relationships.

## Minute 5: Fit OLS Linear Regression

1. Open **Statistical Models**.
2. Choose Linear Regression.
3. Select `target_score` as the outcome.
4. Select predictors such as `sales`, `marketing_spend`, `training_hours`, and `region`.
5. Click the fit button.

Expected result:

- Coefficient table appears.
- Model-level statistics appear.
- Train/test metrics appear.
- A saved ModelRun is created.

## Minute 6: Read Formula And Diagnostics

1. Review the symbolic and estimated formula if available.
2. Inspect residual and diagnostic plots.
3. Read warning text before interpreting coefficients.

Expected result:

- The formula helps explain the model structure.
- Diagnostic plots help identify possible assumption issues.

## Minute 7: Fit ML Baselines

1. Open **Machine Learning**.
2. Choose Regression.
3. Select `target_score` as the target.
4. Select the same predictors.
5. Choose two or three baseline models, such as Linear Regression, Random Forest Regressor, and KNN Regressor.
6. Run the models.

Expected result:

- A results table shows train/test RMSE, MAE, and R-squared.
- Saved ML ModelRuns appear in session state.

## Minute 8: Compare Models

1. Open **Model Comparison**.
2. Filter task type to regression if needed.
3. Compare OLS and ML runs.

Expected result:

- Regression metrics are shown together.
- Classification metrics are not mixed into the regression table.

## Minute 9: Predict A New Observation

1. Open **Prediction**.
2. Select a saved regression model with a readable model label.
3. Enter feature values.
4. Click Predict.

Expected result:

- The app returns a predicted value.
- OLS models may show confidence and prediction intervals.
- ML regression models can optionally show bootstrap empirical intervals.

## Minute 10: Export A Report

1. Open **Report**.
2. Review the report preview.
3. Download Markdown.
4. Download HTML.

Expected result:

- The report includes dataset overview, logs, model summaries, metrics, formulas when available, predictions, and limitations.

## What To Try Next

- Use `sample_data/regression_missing_outliers.csv` to test cleaning, outliers, and VIF diagnostics.
- Use `sample_data/binary_classification.csv` to test logistic regression, ML classification, threshold behavior, and ROC/PR charts.
- Use `sample_data/count_example.csv` to test Poisson and Negative Binomial models.

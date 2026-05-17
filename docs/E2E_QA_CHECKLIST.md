# End-to-End QA Checklist

Last updated: 2026-05-17

Use this checklist for manual QA in the Streamlit app. Start the app, then work through each workflow with the datasets in `sample_data/`.

Recommended command:

```bash
streamlit run app.py
```

General checks for every workflow:

- [ ] The page loads without a red error box.
- [ ] The sidebar shows the expected dataset status.
- [ ] `original_df` remains unchanged after cleaning, transformations, modeling, prediction, and report export.
- [ ] Important actions require user confirmation before changing `working_df`.
- [ ] Empty states and unsupported cases show clear guidance.
- [ ] Notes:

---

## Workflow 1: Basic EDA

Dataset: `sample_data/regression_clean.csv`

Goal: Confirm the app supports a clean exploratory data analysis path.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Open **Upload Data** and upload `regression_clean.csv`. | Upload succeeds; preview appears; original and working copies are created. | [ ] Pass / [ ] Fail |  |
| 2 | Open **Exploratory Data Analysis**. | Page detects the loaded dataset and displays dataset overview. | [ ] Pass / [ ] Fail |  |
| 3 | Inspect the summary table. | Each variable has raw dtype, detected type, missing count, unique count, and relevant numeric/category summaries. | [ ] Pass / [ ] Fail |  |
| 4 | Inspect the missing value table. | Missing counts are displayed; clean dataset should show no major missingness. | [ ] Pass / [ ] Fail |  |
| 5 | Select a numeric variable such as `sales` or `target_score`. | Histogram and boxplot render in chart cards. | [ ] Pass / [ ] Fail |  |
| 6 | Select a categorical variable such as `region`. | Categorical bar chart renders clearly. | [ ] Pass / [ ] Fail |  |
| 7 | Open the correlation matrix section. | Numeric variables can be selected; heatmap and pairwise table render. | [ ] Pass / [ ] Fail |  |
| 8 | Use Relationship Explorer with `sales` and `target_score`. | Scatter plot, Pearson/Spearman summaries, and sample size appear. | [ ] Pass / [ ] Fail |  |
| 9 | Use Relationship Explorer with `region` and `target_score`. | Grouped boxplot and grouped summary table appear. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Workflow 2: Cleaning And Transformations

Dataset: `sample_data/regression_missing_outliers.csv`

Goal: Confirm `working_df` changes only after confirmation, and logs are recorded.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Upload `regression_missing_outliers.csv`. | Upload succeeds and working preview appears. | [ ] Pass / [ ] Fail |  |
| 2 | Open **Data Cleaning**. | Missing table shows missing target/predictor values. | [ ] Pass / [ ] Fail |  |
| 3 | Choose a missing-value handling action, such as median fill for `predictor_a`. | Preview shows the result before applying. | [ ] Pass / [ ] Fail |  |
| 4 | Confirm the cleaning action. | `working_df` updates; `cleaning_log` gains one readable entry. | [ ] Pass / [ ] Fail |  |
| 5 | Open **Transformations**. | Numeric source variables are available. | [ ] Pass / [ ] Fail |  |
| 6 | Create one transformed variable, such as `log1p_predictor_c` or a square transform. | Preview appears first; new column is created only after confirmation. | [ ] Pass / [ ] Fail |  |
| 7 | Check `transformation_log`. | New transformation appears with method, source columns, new column, and notes. | [ ] Pass / [ ] Fail |  |
| 8 | Return to EDA and inspect the new column. | New transformed column appears in `working_df`; original source column remains. | [ ] Pass / [ ] Fail |  |
| 9 | Open **Data Cleaning** and click reset working dataset. | `working_df` is restored from `original_df`; cleaning and transformation logs are cleared. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Workflow 3: Statistical Linear Regression

Dataset: `sample_data/regression_clean.csv`

Goal: Confirm the OLS workflow, diagnostics, prediction interval, and saved run behavior.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Upload `regression_clean.csv`. | Dataset loads successfully. | [ ] Pass / [ ] Fail |  |
| 2 | Open **Statistical Models** and choose Linear Regression. | Continuous numeric target options are shown. | [ ] Pass / [ ] Fail |  |
| 3 | Select `target_score` as y and choose predictors such as `sales`, `marketing_spend`, `training_hours`, and `region`. | Predictors can include numeric and categorical variables. | [ ] Pass / [ ] Fail |  |
| 4 | Fit OLS. | Model fits; no unexpected red error appears. | [ ] Pass / [ ] Fail |  |
| 5 | Inspect coefficient table. | Terms, estimates, standard errors, t values, p-values, and confidence intervals appear. | [ ] Pass / [ ] Fail |  |
| 6 | Inspect LaTeX formula. | Symbolic and/or estimated formula displays clearly. | [ ] Pass / [ ] Fail |  |
| 7 | Inspect diagnostics. | Residual, Q-Q, scale-location, leverage/Cook's distance or influence outputs appear where available. | [ ] Pass / [ ] Fail |  |
| 8 | Open **Model Diagnostics** for the saved OLS run. | Metadata, overfitting diagnostics, VIF/multicollinearity diagnostics, and influence diagnostics appear. | [ ] Pass / [ ] Fail |  |
| 9 | Open **Prediction** and select the OLS run. | Saved run label is readable; model metadata/formula appears. | [ ] Pass / [ ] Fail |  |
| 10 | Enter one new observation and predict. | Point prediction plus mean confidence interval and observation prediction interval appear. | [ ] Pass / [ ] Fail |  |
| 11 | Open **Model Comparison**. | OLS run appears in the regression table with AIC/BIC/adjusted R-squared where available. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Workflow 4: Logistic Regression

Dataset: `sample_data/binary_classification.csv`

Goal: Confirm positive-class handling, threshold behavior, classification plots, and prediction.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Upload `binary_classification.csv`. | Dataset loads successfully. | [ ] Pass / [ ] Fail |  |
| 2 | Open **Statistical Models** and choose Binary Logistic Regression. | Binary target options are shown. | [ ] Pass / [ ] Fail |  |
| 3 | Select `purchased` as the target. | Positive class selector appears. | [ ] Pass / [ ] Fail |  |
| 4 | Choose positive class `yes`. | Page records `yes` as the positive class. | [ ] Pass / [ ] Fail |  |
| 5 | Select predictors such as `age`, `income_score`, `visits`, and `channel`. | Predictors can include numeric and categorical variables. | [ ] Pass / [ ] Fail |  |
| 6 | Fit logistic regression. | Model fits or shows a clear warning if separation/convergence issues occur. | [ ] Pass / [ ] Fail |  |
| 7 | Inspect odds ratios. | Odds ratio and odds-ratio confidence interval columns appear. | [ ] Pass / [ ] Fail |  |
| 8 | Inspect ROC, PR, and confusion matrix. | Charts/tables render and are labeled with the positive class/threshold context. | [ ] Pass / [ ] Fail |  |
| 9 | Adjust threshold or choose F1 optimization. | Confusion matrix and displayed metrics change with the threshold. | [ ] Pass / [ ] Fail |  |
| 10 | Open **Prediction** and select the saved logistic run. | Metadata and formula appear if available. | [ ] Pass / [ ] Fail |  |
| 11 | Enter one new observation and predict. | Positive-class probability, threshold, predicted class, and class probabilities appear. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Workflow 5: ML Baseline

Dataset options:

- Regression: `sample_data/regression_clean.csv`
- Binary classification: `sample_data/binary_classification.csv`

Goal: Confirm baseline sklearn Pipelines, metrics, interpretability outputs, and prediction.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Upload `regression_clean.csv`. | Dataset loads successfully. | [ ] Pass / [ ] Fail |  |
| 2 | Open **Machine Learning** and choose Regression. | Numeric targets are available. | [ ] Pass / [ ] Fail |  |
| 3 | Select `target_score` and predictors. | Model controls appear, including split, random state, scaling, CV/tuning options where available. | [ ] Pass / [ ] Fail |  |
| 4 | Fit at least three models, such as Linear Regression, Random Forest Regressor, and KNN Regressor. | Results table shows train/test RMSE, MAE, and R-squared for each model. | [ ] Pass / [ ] Fail |  |
| 5 | Inspect feature importance for a tree-based model. | Feature importance table and chart appear for Random Forest or Decision Tree. | [ ] Pass / [ ] Fail |  |
| 6 | Inspect overfitting warnings in **Model Diagnostics**. | Train/test gaps and any warning cards are visible. | [ ] Pass / [ ] Fail |  |
| 7 | Open **Prediction** and select one ML regression run. | Prediction Console uses saved fitted artifact and shows readable model metadata. | [ ] Pass / [ ] Fail |  |
| 8 | Upload `binary_classification.csv`. | Dataset loads and previous workflow behavior remains stable. | [ ] Pass / [ ] Fail |  |
| 9 | Open **Machine Learning** and choose Binary Classification. | Binary target options and positive class selector appear. | [ ] Pass / [ ] Fail |  |
| 10 | Fit at least three models, such as Logistic Regression, Random Forest Classifier, and KNN Classifier. | Results table shows accuracy, precision, recall, F1, ROC AUC, and PR AUC where feasible. | [ ] Pass / [ ] Fail |  |
| 11 | Inspect classification feature importance where available. | Tree-based importance appears for supported models. | [ ] Pass / [ ] Fail |  |
| 12 | Open **Prediction** and predict with one ML classification run. | Positive-class probability, threshold, predicted class, and class probabilities appear. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Workflow 6: Report Export

Suggested setup:

- upload `regression_clean.csv`
- run at least one EDA chart
- apply at least one cleaning or transformation if using a dataset with missing values
- fit at least one statistical model
- fit at least one ML model
- make at least one prediction

Goal: Confirm Markdown and HTML reports summarize the current analysis honestly.

| Step | Action | Expected result | Pass/Fail | Notes |
|---:|---|---|---|---|
| 1 | Complete at least one analysis workflow with saved model runs. | Session has dataset, logs or empty log sections, and at least one ModelRun. | [ ] Pass / [ ] Fail |  |
| 2 | If chart/report actions are available, add selected charts/results to report. | Report-related selections do not break page rendering. | [ ] Pass / [ ] Fail |  |
| 3 | Open **Report**. | Report preview appears without red errors. | [ ] Pass / [ ] Fail |  |
| 4 | Confirm dataset overview. | Rows, columns, and column names are included. | [ ] Pass / [ ] Fail |  |
| 5 | Confirm cleaning and transformation logs. | Logs appear when available; honest empty state appears when none exist. | [ ] Pass / [ ] Fail |  |
| 6 | Confirm model summaries. | Model name, family, task type, target, features, split config, preprocessing, metrics, CV metrics where available, AIC/BIC where available appear. | [ ] Pass / [ ] Fail |  |
| 7 | Confirm formulas/coefficient tables where available. | Statistical formulas and coefficient tables appear; ML coefficients are labeled as predictive, not inferential. | [ ] Pass / [ ] Fail |  |
| 8 | Confirm prediction examples and intervals where available. | Saved prediction examples appear; statistical/ML intervals are labeled by method. | [ ] Pass / [ ] Fail |  |
| 9 | Confirm limitations section. | Report includes limitations about causality, feature importance, assumptions, intervals, and export scope. | [ ] Pass / [ ] Fail |  |
| 10 | Download Markdown report. | `.md` download works and content is readable. | [ ] Pass / [ ] Fail |  |
| 11 | Download HTML report. | `.html` download works and opens/readable in browser. | [ ] Pass / [ ] Fail |  |

Workflow result:

- [ ] Pass
- [ ] Fail

Notes:

---

## Final QA Sign-off

| Area | Pass/Fail | Notes |
|---|---|---|
| Basic EDA | [ ] Pass / [ ] Fail |  |
| Cleaning and transformations | [ ] Pass / [ ] Fail |  |
| Statistical linear regression | [ ] Pass / [ ] Fail |  |
| Logistic regression | [ ] Pass / [ ] Fail |  |
| ML baseline | [ ] Pass / [ ] Fail |  |
| Report export | [ ] Pass / [ ] Fail |  |

Overall result:

- [ ] Ready for broader manual testing
- [ ] Needs fixes before broader testing

General notes:


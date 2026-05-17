# UI/UX Guide

This guide defines the project-wide user experience standard for Stats Learning Lab. Use it before changing page layout, page copy, charts, tooltips, or educational guidance.

## Product Identity

Stats Learning Lab is a modest learning-first app for practicing applied statistics and machine learning workflows.

It should feel:

- educational
- professional
- transparent
- calm
- beginner-friendly without hiding important assumptions
- guided, but not black-box

The app should never feel like AutoML. The user should choose the dataset, target, features, cleaning operations, transformations, models, preprocessing options, and interpretation context.

At every point, the user should know:

- where they are in the workflow
- what this page is for
- what inputs they need to provide
- what the app will do after they click an action button
- what the app will not do automatically
- what the next recommended step is

## Visual Style

Use a clean academic/professional look.

Preferred style:

- light theme by default
- calm primary color, such as muted blue, teal, or slate
- neutral page backgrounds
- clear text contrast
- card-based sections for repeated or self-contained content
- generous spacing between sections
- restrained typography
- no decorative clutter
- no ornamental backgrounds
- no visual effects that distract from analysis

The interface should resemble a careful research notebook or professional analytics workbench, not a marketing landing page and not a dense developer console.

## Page Structure Standard

Every major page should follow this structure where feasible:

1. Page title
2. Short purpose statement
3. "When to use this page"
4. "What this page does not do"
5. Main controls
6. Results area
7. Interpretation notes
8. Next recommended step

Recommended page skeleton:

```text
Page title
Short purpose statement

Guidance panel
- When to use this page
- What this page does not do
- Next recommended step

Main controls
Results
Interpretation notes
```

For crowded pages, use tabs or expanders to keep the page readable. Do not hide primary controls inside expanders. Use expanders for advanced settings, raw tables, technical details, and long logs.

## Homepage Structure

The homepage should explain the product clearly and help the user start.

Recommended homepage sections:

1. Hero section
   - Product name
   - One-sentence explanation
   - Clear quick start instruction

2. Workflow overview
   - Upload
   - Explore
   - Clean
   - Transform
   - Model
   - Compare
   - Predict
   - Diagnose
   - Report

3. Module cards
   - EDA
   - Data Cleaning
   - Transformations
   - Statistical Models
   - Machine Learning
   - Model Comparison
   - Prediction
   - Diagnostics
   - Report

4. Example outputs
   - summary table
   - missingness chart
   - model comparison table
   - prediction result
   - report export

5. What the app helps automate
   - schema detection
   - summaries
   - chart generation
   - metric calculation
   - model result storage
   - report assembly

6. What the user still controls
   - target variable
   - feature variables
   - cleaning choices
   - transformations
   - model family
   - model options
   - interpretation context

7. Quick start
   - "Go to Upload Data"
   - or short instructions using the sidebar

The homepage must not claim that the app automatically finds the best model or automatically proves causal effects.

## Glossary And Tooltip Standard

Important statistical and machine learning terms should have a small help cue. Prefer Streamlit native `help=` text where available.

Use:

- short hover explanations for compact controls
- longer explanations in popovers or expanders when the concept is subtle
- examples when useful
- warnings for commonly misunderstood terms

Terms that should receive tooltip or glossary coverage:

- target variable
- feature / predictor
- train/test split
- random state
- scaling
- imputation
- one-hot encoding
- cross-validation
- hyperparameter tuning
- p-value
- confidence interval
- prediction interval
- odds ratio
- ROC AUC
- PR AUC
- F1 score
- RMSE
- MAE
- R-squared
- AIC
- BIC
- VIF
- Cook's distance
- leverage
- residual
- outlier
- anomaly
- calibration
- PDP
- ICE
- feature importance
- permutation importance
- bootstrap interval
- Box-Cox
- Yeo-Johnson

Tooltip writing rules:

- Use one or two plain-language sentences.
- Avoid equations unless the control is explicitly mathematical.
- Include the practical implication.
- Say when a metric is descriptive, predictive, or inferential.
- Warn when something does not imply causality.

Example:

```text
ROC AUC: Measures how well a binary classifier ranks positive cases above negative cases across thresholds. It does not tell you which threshold is best for your use case.
```

## Chart Presentation Standard

Every important chart should appear in a chart card or chart-like section.

Each chart card should include:

- chart title
- variables used
- method used
- sample size used after dropping unavailable values
- chart itself
- "How to read this chart"
- warnings or limitations
- optional "Add to report" button when report selection is implemented

Recommended chart card structure:

```text
Chart title
Variables: y vs x
Method: Pearson correlation / OLS residual plot / PCA projection / etc.
Rows used: n

[chart]

How to read this chart:
...

Limitations:
...
```

Chart rules:

- Do not rely only on color to communicate meaning.
- Prefer clear axis titles over raw variable names when practical.
- Use consistent color semantics across pages.
- Keep charts uncrowded.
- Use tables or download buttons for large data details.
- Put long technical details below the chart or in an expander.
- Explain whether a chart is exploratory, diagnostic, inferential, or predictive.

Common warnings:

- Correlation does not imply causation.
- Feature importance does not imply causation.
- PDP and ICE show model behavior, not causal effects.
- Outlier or anomaly flags do not prove rows are errors.
- ML uncertainty intervals are empirical unless explicitly statistical.

## Button Language

Buttons should be action-oriented and specific.

Preferred labels:

- "Run analysis"
- "Preview cleaning operation"
- "Apply cleaning operation"
- "Preview transformation"
- "Create transformed variable"
- "Fit model"
- "Run baseline models"
- "Save model run"
- "Compute diagnostics"
- "Generate prediction"
- "Add to report"
- "Download report"
- "Reset working data"

Avoid vague labels such as:

- "Submit"
- "Go"
- "OK"
- "Process"
- "Run" without context

Use destructive or high-impact labels carefully:

- "Clear saved model runs"
- "Reset working data from original dataset"

High-impact actions should have clear nearby text explaining what will change.

## Error And Empty State Standard

Empty states and errors should be friendly, specific, and actionable.

### No Dataset Loaded

Use when `working_df` is unavailable.

Recommended message:

```text
No dataset is loaded yet. Upload a CSV or Excel file on the Upload Data page to begin.
```

Next step:

```text
Next: go to Upload Data.
```

### No Model Selected

Recommended message:

```text
Choose at least one model before running this analysis.
```

Next step:

```text
Select one or more model checkboxes, then run the analysis.
```

### No Model Runs Saved

Recommended message:

```text
No saved model runs are available yet. Fit a model from Statistical Models or Machine Learning first.
```

Next step:

```text
Next: fit and save a model run.
```

### Not Enough Data

Recommended message:

```text
There is not enough complete data for this operation after missing values are removed.
```

Next step:

```text
Try fewer variables, inspect missing values, or clean the working dataset first.
```

### Unsupported Variable Type

Recommended message:

```text
This variable type is not supported for the selected method.
```

Next step:

```text
Choose a compatible numeric, binary, categorical, datetime, or count-like variable.
```

Error writing rules:

- Start with the problem in plain language.
- Explain why it happened when known.
- Give a concrete next step.
- Avoid Python traceback language in user-facing text unless debugging is explicitly requested.
- Keep raw exceptions available only in a technical expander if needed.

## Accessibility

Accessibility expectations:

- Do not rely only on color.
- Use readable labels.
- Avoid very small text.
- Keep chart legends clear.
- Avoid over-crowded charts.
- Add captions and warnings.
- Use high contrast for important statuses.
- Long tables should be collapsible, paginated by Streamlit, or downloadable.
- Keep action labels unique and descriptive.
- Avoid making users remember information from earlier sections.
- Put critical assumptions close to the controls they affect.

For status indicators, combine color with text:

- "Available"
- "Warning"
- "Not available"
- "Needs review"
- "Saved"

## Implementation Constraints

Prefer Streamlit native layout elements:

- `st.tabs`
- `st.columns`
- `st.expander`
- `st.container`
- `st.popover` where available
- `help=` on inputs
- `st.caption`
- `st.info`, `st.warning`, `st.error`, `st.success`

Use limited custom CSS:

- only for spacing, card styling, and light visual polish
- keep selectors simple
- avoid fragile Streamlit internal class selectors where possible

Do not use unsafe JavaScript.

Do not pass user-uploaded data directly into custom HTML.

Do not change model logic, data logic, train/test behavior, preprocessing behavior, or session-state mutation rules while doing UI refactors.

Keep UI helpers separate from core logic. A future UI helper layer may live under:

```text
src/ui/
```

Potential helpers:

- page header
- guidance panel
- metric cards
- chart card
- empty state
- glossary/help text
- download button wrapper
- model run selector
- preview/confirm panel

## Refactor Priority

Recommended UI implementation order:

1. Update homepage copy and workflow overview.
2. Add reusable UI helpers for page headers, guidance panels, empty states, and chart cards.
3. Apply the page structure standard to Upload Data, Data Cleaning, Transformations, Report, and Prediction first.
4. Reorganize EDA into clearer sections or tabs.
5. Reorganize Statistical Models into consistent configure/results/interpretation/diagnostics sections.
6. Reorganize Machine Learning into clearer task sections and move advanced interpretation into a focused area.
7. Improve Model Comparison and Model Diagnostics with clearer ranking, risk, and next-step guidance.
8. Add glossary/tooltips gradually as pages are touched.
9. Run tests and a manual Streamlit walkthrough after each group of UI changes.

## Non-Goals

Do not use the UI refresh to:

- change modeling behavior
- add new model families
- add black-box AutoML
- alter `original_df` or `working_df` rules
- remove user choice from modeling workflows
- make causal claims
- persist fitted model objects to disk

The UI should make the current transparent workflow easier to understand, not replace it with hidden automation.

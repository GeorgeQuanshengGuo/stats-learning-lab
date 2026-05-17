# Troubleshooting

This guide lists common issues and practical fixes.

## Streamlit Will Not Start

### Symptom

The terminal says `streamlit: command not found`.

### Fix

Activate your virtual environment and install requirements:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## `python` Command Not Found

### Symptom

The terminal says `python: command not found`.

### Fix

Use `python3`:

```bash
python3 -m pytest
python3 -m venv .venv
```

On Windows, `python` usually works after installing Python from python.org or the Microsoft Store.

## Package Installation Fails

### Common causes

- outdated `pip`
- wrong Python version
- virtual environment not activated
- network or package index issue

### Fix

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

If Excel upload fails, confirm that `openpyxl` is installed.

## Port Already In Use

### Symptom

Streamlit says port `8501` is already in use.

### Fix

Run on another port:

```bash
streamlit run app.py --server.port 8502
```

## Uploaded File Does Not Load

### Common causes

- unsupported file type
- malformed CSV
- Excel dependency missing
- file is open or locked by another program

### Fix

- Use `.csv`, `.xlsx`, or `.xls`.
- Try opening the file in a spreadsheet app and saving a fresh CSV.
- For Excel files, install requirements again.

## Data Types Look Wrong

### Symptom

A numeric column is detected as categorical or text.

### Possible causes

- numbers contain commas or currency symbols
- missing values use custom strings such as `N/A`, `unknown`, or `-`
- mixed numeric and text values exist in the same column

### Fix

Use EDA and cleaning steps to inspect the column. If needed, clean the source file before upload. The app has type detection, but it cannot always infer the intended meaning of messy columns.

## Model Fitting Fails

### Common causes

- too few rows
- target has missing values
- target has only one class
- too many predictors for the number of rows
- perfect separation in logistic regression
- singular matrix from highly correlated predictors
- unsupported target type for the selected model

### Fix

1. Check EDA and missing values.
2. Reduce the number of features.
3. Remove or combine rare categories if appropriate.
4. Use VIF and correlation diagnostics for highly correlated predictors.
5. For binary targets, confirm that both classes appear in train and test splits.
6. Try an ML baseline if a statistical model is too unstable, but do not treat that as inference.

## Logistic Regression Warnings

### Symptom

Warnings about convergence, overflow, or perfect separation.

### Meaning

The data may separate the classes too perfectly, or there may be too few observations in one class.

### Fix

- Check class balance.
- Reduce features.
- Combine sparse categories.
- Consider ML classification for prediction.
- Avoid interpreting unstable odds ratios.

## Cross-Validation Or Tuning Is Slow

### Cause

Cross-validation, GridSearchCV, RandomizedSearchCV, permutation importance, PDP/ICE, bootstrap intervals, and learning curves can refit many models.

### Fix

- Use fewer models.
- Use fewer CV folds.
- Use RandomizedSearchCV instead of GridSearchCV.
- Turn off permutation importance or bootstrap intervals.
- Start with a small sample dataset.

## Prediction Page Shows No Usable Models

### Cause

Prediction requires a saved fitted model artifact. Old ModelRun metadata without an artifact can appear in comparison but cannot predict.

### Fix

Refit the model in the current session, then return to Prediction.

## Prediction Input Fails

### Common causes

- missing required feature value
- value has incompatible type
- category was not seen during fitting
- restored artifact is unavailable or incompatible

### Fix

Check the model metadata shown on the Prediction page. Enter raw values for every feature listed. For sklearn pipelines, unknown categories are handled where possible, but some models or artifacts can still fail on incompatible inputs.

## Report Export Fails Or Looks Empty

### Common causes

- no dataset loaded
- no model runs saved
- no predictions made
- report section has no available data yet

### Fix

Run at least one workflow:

1. upload a dataset
2. inspect EDA
3. fit one model
4. optionally make one prediction
5. return to Report

The report is honest about empty sections; it does not invent unavailable results.

## Workspace Restore Questions

### Important

Workspace snapshots use local pickle files. Restore only snapshots created by this app on your own machine.

### If restore behaves unexpectedly

Use **Start fresh** from the sidebar, then upload the dataset again.

## Red Error Box In The App

If you see a red error box:

1. Read the top error message.
2. Note which page and action caused it.
3. Try the same workflow with a sample dataset.
4. If the error repeats, use the traceback and the sample dataset to debug.

Good first validation command:

```bash
python3 -m pytest
```

# Report Reproducibility and Completeness Audit

Last updated: 2026-05-17

## Scope

This audit checks whether exported Markdown and HTML reports contain enough information for a user to understand and broadly reproduce the analysis choices made in the current Streamlit session.

Reviewed files:

- `src/reporting/report_builder.py`
- `src/reporting/export_markdown.py`
- `src/reporting/export_html.py`
- `pages/08_report.py`
- `tests/test_report_builder.py`
- `tests/test_report_reproducibility.py`

## Summary Finding

The report export now includes the key reproducibility components expected for the current app:

- generated timestamp
- dataset overview
- working dataset status
- schema summary
- missing value summary
- cleaning log
- transformation log
- model run summaries
- selected target and features
- model family, model name, and task type
- train/test split configuration
- preprocessing summary
- model formula and estimated formula where available
- coefficient table where available
- train/test metrics
- CV metrics where available
- compact diagnostics summary
- prediction examples and interval output where available
- task-specific model comparison tables
- interval/coefficient interpretation notes
- limitations section

The report continues to avoid unsupported causal claims and includes explicit limitation language around descriptive EDA, feature importance, ML coefficients, and interval assumptions.

## Checks And Current Behavior

| Check | Current behavior | Status |
|---|---|---|
| dataset overview | reports row count, column count, and column names | implemented |
| working dataset status | reports `working_df` role, rows, columns, cleaning step count, transformation step count, and immutability note | implemented |
| schema summary | uses supplied schema or builds current summary table | implemented |
| missing value summary | uses supplied missing table or builds current missing table | implemented |
| cleaning log | includes confirmed cleaning operations or honest empty text | implemented |
| transformation log | includes confirmed transformation operations or honest empty text | implemented |
| selected target and features | included in each saved model run summary | implemented |
| model type | includes model name, family, and task type | implemented |
| train/test split | included as `split_config` table | implemented |
| preprocessing summary | included as table for each model run | implemented |
| model formula | included when saved in ModelRun/artifact | implemented |
| coefficient table | included when saved; missing table is clearly stated | implemented |
| train/test metrics | included for every saved run with metrics | implemented |
| diagnostics | includes diagnostic plot keys, overfitting/CV diagnostics, influence metadata where available | implemented |
| predictions | prediction examples and intervals are included from `prediction_log` | implemented |
| model comparison | task-specific comparison tables are included | implemented |
| limitations | report includes limitations and avoids causal overclaims | implemented |
| generated timestamp | included near top of Markdown and HTML reports | implemented |

## Fixes Made

### Working Dataset Status

Added `working_dataset_status` to the report context.

This records:

- dataset role: `working_df`
- row count
- column count
- number of cleaning steps
- number of transformation steps
- note that the original uploaded dataset is kept separately and is not modified

Files updated:

- `src/reporting/report_builder.py`
- `src/reporting/export_markdown.py`
- `src/reporting/export_html.py`

### Diagnostics Summary

Added a compact per-model diagnostics summary.

The report can now include:

- saved diagnostic plot keys
- overfitting / CV stability diagnostics derived from ModelRun metrics
- influence row count and top influential rows when an artifact provides influence metadata
- decision tree explanation warning when available

Files updated:

- `src/reporting/report_builder.py`
- `src/reporting/export_markdown.py`
- `src/reporting/export_html.py`

### Interval And Coefficient Notes

Added a global report section for interval and coefficient interpretation.

The report now distinguishes:

- confidence intervals
- prediction intervals
- ML empirical uncertainty intervals
- statistical inference coefficients
- predictive ML coefficients

Files updated:

- `src/reporting/report_builder.py`
- `src/reporting/export_markdown.py`
- `src/reporting/export_html.py`

## Causal Claims Audit

The report limitation text states that exploratory summaries are descriptive and do not establish causality.

The report also states:

- feature importance is a predictive diagnostic and does not establish causality
- machine learning coefficients may be on a transformed or standardized feature scale
- unsupported statistical interpretations are not generated automatically

This is consistent with the project principle that the app should be transparent and not black-box.

## Tests Added

New file:

- `tests/test_report_reproducibility.py`

The tests verify that:

- report context contains dataset overview, working status, schema, missing summary, logs, model summaries, model comparison, timestamp, notes, and limitations
- model summaries contain target, features, split configuration, preprocessing, formulas, coefficient tables, metrics, diagnostics, and prediction intervals
- Markdown export includes all major reproducibility sections and caveats
- HTML export includes the same major sections
- empty reports remain honest when no model runs are available

## Validation Results

Targeted command:

```bash
python3 -m pytest tests/test_report_reproducibility.py tests/test_report_builder.py
```

Result:

- 13 passed

Full command:

```bash
python3 -m pytest
```

Result:

- 384 passed
- 1 warning from `joblib/loky` about physical CPU core detection during clustering tests

## Sample Workflow Report Generation

A report-like sample workflow was validated programmatically in `tests/test_report_reproducibility.py`.

The synthetic context included:

- working dataset
- cleaning log
- transformation log
- statistical OLS ModelRun
- ML regression ModelRun
- formulas
- coefficient tables
- train/test metrics
- CV metrics
- influence diagnostics metadata
- statistical prediction interval
- ML bootstrap empirical interval
- Markdown export
- HTML export

Manual browser-based report export should still be included in end-to-end QA using `docs/E2E_QA_CHECKLIST.md`.

## Remaining Limitations

- Reports summarize the current Streamlit session; they are not a complete executable analysis script.
- Fitted model objects are not exported into reports.
- Chart image embedding remains limited; report text includes available chart metadata and saved outputs where present.
- Prediction examples appear only when the user has generated predictions during the session.
- Diagnostic depth depends on what each model run/artifact saved.
- HTML/Markdown exports do not replace formal statistical review of assumptions, study design, or causal identification.

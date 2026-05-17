# Edge Case Testing

Last updated: 2026-05-17

## Scope

This document records robustness checks for messy real-world datasets and unsupported modeling situations.

New test file:

- `tests/test_edge_cases.py`

The goal is not to force every method to work on every dataset. The goal is to ensure the app either:

- produces a reasonable result,
- returns a clear warning,
- or fails with a clear message instead of silently producing misleading output.

## Tested Edge Cases

| Edge case | Expected behavior | Current behavior |
|---|---|---|
| all-missing column | detected safely, no crash in summary | detected as `constant`; summary records full missing count |
| constant column | detected safely, no misleading numeric stats | detected as `constant` |
| ID-like column | detected as identifier-like when name and uniqueness suggest it | detected as `id_like` |
| duplicate CSV column names | loader should not crash | pandas disambiguates as `value`, `value.1`, etc. |
| non-ASCII column names | EDA summaries should preserve names | tested with `日期`; handled correctly |
| column names with spaces and punctuation | summaries should preserve names | tested with `score value (%)`; handled correctly |
| tiny dataset | model fitting should fail clearly when split is impossible | ML regression raises a clear `At least 3 rows` message |
| target with only one class | classification should fail clearly | ML binary classification raises a clear two-class validation error |
| severe class imbalance | model can fit but should warn about unstable metrics | ML binary classification adds warning text to ModelRun notes |
| logistic regression perfect separation | should warn instead of looking like a normal stable fit | statsmodels warning is captured in result and shown on the statistical models page |
| high-cardinality categorical variable | one-hot encoder should not crash and should handle unknown levels | preprocessing transforms train/test safely with `handle_unknown="ignore"` |
| numeric column stored as string | should be detected as numeric when most values parse | type detector now classifies numeric strings as numeric after datetime/ordinal checks |
| date-like column | should be detected as datetime when parsing success is high | detected as `datetime` |
| extreme outliers | outlier module should flag, not delete | IQR method flags the extreme row and does not modify input |
| missing target values | modeling should drop missing targets or fail clearly | ML regression drops missing target rows and records `rows_used` |
| more predictors than rows | regularized/pipeline ML path should not crash automatically | Ridge baseline fits and returns metrics; interpretation remains limited |
| VIF with singular matrix | should not crash; should flag multicollinearity | VIF returns infinite values with strong warnings; condition number warns |
| prediction with missing input values | pipelines with imputation should handle missing raw values | prediction succeeds through saved fitted Pipeline |
| model run without saved artifact | prediction should fail clearly | raises `PredictionError` with a clear message |
| report generation with no model runs | report should include honest empty sections | report context returns empty model summaries/comparison tables |

## Fixes Made

### Numeric Strings

Updated `src/data/type_detector.py` so columns that arrive as strings but are mostly numeric are classified as numeric after datetime and ordinal checks.

Why:

- real CSV files often store numeric values as text
- treating these as nominal categories can mislead EDA and downstream choices

### Logistic Perfect Separation Warnings

Updated `src/modeling/statistical/logistic_regression.py` to capture statsmodels fit warnings and return them in the result dictionary.

Updated `pages/05_statistical_models.py` to display those warnings after fitting binary logistic regression.

Why:

- perfect separation can produce unstable or unidentified parameters
- users should see this warning in the app instead of only in the console

### Extreme Odds Ratio Overflow

Updated logistic coefficient table generation to suppress raw numpy overflow warnings while still allowing infinite odds ratios to appear when coefficients are extreme.

Why:

- infinite odds ratios are meaningful diagnostic evidence of instability
- raw runtime warnings are noisy and less beginner-friendly

### Severe Class Imbalance Notes

Updated `src/modeling/machine_learning/classification.py` so binary ML ModelRuns include notes when train/test splits are severely imbalanced or when a split contains only one class.

Why:

- stratified splitting can still produce a one-class test split when the minority class has very few rows
- metrics such as ROC AUC may be undefined or unstable in that situation

### VIF Singular Matrix Warning Noise

Updated `src/modeling/diagnostics/multicollinearity.py` to suppress low-level divide-by-zero runtime warnings while returning infinite VIF values and strong warning messages.

Why:

- infinite VIF is the useful result
- low-level numerical warnings should not crowd the user-facing workflow

## Validation Results

Targeted command:

```bash
python3 -m pytest tests/test_edge_cases.py
```

Result:

- 14 passed

Full validation command:

```bash
python3 -m pytest
```

Result:

- 367 passed
- 1 warning from `joblib/loky` about physical CPU core detection during clustering tests

## Remaining Limitations

- Duplicate column behavior is validated through CSV upload parsing, where pandas disambiguates names automatically. Manually constructed DataFrames with true duplicate column labels remain an unsupported internal edge case.
- Perfect separation is warned about, but the app does not automatically switch to penalized logistic regression or exact logistic regression.
- Severe class imbalance is warned about, but the app does not yet implement resampling, class weighting UI, calibration workflows for imbalance, or imbalance-specific model selection.
- More predictors than rows can be fitted by some ML models, but statistical inference in such designs may be unstable or not meaningful.
- High-cardinality categorical variables can be encoded, but they can create many sparse features and may be slow or hard to interpret.
- Missing prediction input values are handled when the saved Pipeline has imputation. Models without appropriate preprocessing may still reject missing inputs.

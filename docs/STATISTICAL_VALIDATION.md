# Statistical Validation

Last updated: 2026-05-17

This document records the statistical correctness checks added during stabilization. The goal is to verify that core numeric outputs are consistent with trusted libraries such as statsmodels, scipy, pandas, and scikit-learn.

## Validation Scope

Added tests:

```text
tests/test_statistical_correctness.py
```

These tests use small synthetic datasets where outputs can be checked directly against trusted library calculations. They compare numeric values with tolerances rather than comparing formatted strings.

## What Was Tested

### OLS Linear Regression

Validated against direct statsmodels OLS results:

- coefficient estimates
- standard errors
- t values
- p-values
- confidence interval bounds
- R-squared
- adjusted R-squared
- F-statistic
- AIC
- BIC

Validated prediction interval helper against `statsmodels` `get_prediction().summary_frame()`:

- predicted mean
- mean confidence interval lower/upper bounds
- observation prediction interval lower/upper bounds
- prediction interval wider than mean confidence interval on tested rows

### Logistic Regression

Validated against direct statsmodels Logit results:

- coefficient estimates
- standard errors
- z values
- p-values
- odds ratios
- AIC
- BIC
- pseudo R-squared
- predicted probabilities

Validated classification metrics against scikit-learn:

- accuracy
- precision
- recall
- F1
- ROC AUC

### Correlation

Validated against pandas and scipy:

- Pearson correlation matrix
- Spearman correlation
- pairwise correlation table ordering and values
- high-correlation pair detection, including positive and negative direction labels

### VIF

Validated against statsmodels `variance_inflation_factor`:

- VIF values for correlated numeric predictors
- VIF warning thresholds:
  - `<= 5`: ok
  - `> 5`: warning
  - `> 10`: strong warning

### Influence Diagnostics

Validated against statsmodels `OLSInfluence`:

- leverage / hat values
- Cook's distance
- internally standardized residuals
- influence table flag columns
- helper filters for large residuals, high leverage, and large Cook's distance

### Missing Value Summaries

Validated against direct pandas missing-value counts:

- variable-level missing counts
- variable-level missing percentages
- row-level missing counts
- row-level missing percentages

## Commands Run

Targeted correctness tests:

```bash
python3 -m pytest tests/test_statistical_correctness.py
```

Result:

```text
10 passed
```

Full test suite:

```bash
python3 -m pytest
```

Result after adding correctness tests:

```text
338 passed, 1 warning
```

The warning is the existing joblib/loky CPU-core detection warning during clustering tests. It does not indicate a statistical validation failure.

## What Passed

- Core OLS inference tables match statsmodels numeric outputs.
- OLS model-level statistics match statsmodels numeric outputs.
- OLS prediction interval helper matches statsmodels summary frames.
- Core Logit inference tables match statsmodels numeric outputs.
- Logit odds ratios are correctly computed as `exp(coef)`.
- Logit predicted probabilities are in `[0, 1]` and match statsmodels prediction output.
- Binary classification metrics match scikit-learn for the tested data.
- Correlation helpers match pandas/scipy for the tested data.
- VIF helper matches statsmodels VIF for the tested numeric design matrix.
- Influence diagnostics match statsmodels OLSInfluence for leverage, Cook's distance, and standardized residuals.
- Missing-value summaries match pandas counts and percentages.

## What Failed

No statistical correctness failures were found in this validation pass.

No analytics logic changes were required.

## Remaining Unvalidated Areas

The correctness tests are intentionally focused. The following areas still need future validation before treating them as statistically hardened:

- Multinomial logistic regression coefficient tables and predicted probabilities against direct statsmodels MNLogit reference outputs.
- Ordinal regression thresholds, coefficients, probabilities, and metrics against direct `OrderedModel` references.
- Poisson and Negative Binomial coefficients, incidence rate ratios, deviance, Pearson chi-square, and expected count prediction intervals against direct statsmodels references.
- Target transformation inverse-scale metrics for transformed OLS targets.
- Robustness under singular design matrices, near-perfect multicollinearity, perfect logistic separation, rare events, and sparse categorical levels.
- PR AUC implementation details across edge cases; binary core metrics and ROC AUC are now validated, but PR AUC was not included in this numerical comparison pass.
- Calibration curve binning against sklearn references.
- PDP/ICE numeric output against sklearn inspection references.
- Bootstrap ML uncertainty interval behavior beyond basic existing tests.
- Report-level statistical summaries beyond verifying report context construction.

## Recommended Next Validation Steps

1. Add direct statsmodels-reference tests for count, multinomial, and ordinal models.
2. Add edge-case tests for singular matrices, separation, all-one-class splits, and small sample sizes.
3. Add manual end-to-end runs using `sample_data/` datasets and record results in `docs/VALIDATION_STATUS.md`.
4. Keep all statistical tests numeric and tolerant, not string-format based.

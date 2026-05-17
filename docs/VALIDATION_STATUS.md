# Validation Status

Validation was last updated on 2026-05-17 in:

```text
<project-root>
```

## Latest Commands Run

```bash
python3 -m pytest
```

Result: passed.

```text
372 passed, 1 warning in 5.43s
```

The warning came from joblib/loky while it tried to detect physical CPU cores during ML regression tests. It did not fail the suite.

```bash
git status --short
```

Result: passed. No uncommitted files were reported during the release readiness check.

```text
<clean working tree>
```

## Python Command Note

Earlier validation showed that this local environment does not provide a `python` command:

```text
zsh:1: command not found: python
```

Use `python3` unless a `python` alias is created.

## Streamlit Smoke Status

The Streamlit app was smoke-tested successfully when local port binding was allowed:

```bash
streamlit run app.py --server.headless true --server.port 8510
curl -I http://localhost:8510/machine_learning
```

The local request returned:

```text
HTTP/1.1 200 OK
```

After adding clustering, the same smoke check was repeated on port 8511 and `/machine_learning` also returned:

```text
HTTP/1.1 200 OK
```

After adding anomaly detection, the same smoke check was repeated on port 8512 and `/machine_learning` also returned:

```text
HTTP/1.1 200 OK
```

After the latest stabilization and documentation pass, a temporary Streamlit instance was started on a separate local port and the main routes were checked. These routes returned `200 OK`:

- `/`
- `/upload_data`
- `/eda`
- `/data_cleaning`
- `/transformations`
- `/statistical_models`
- `/machine_learning`
- `/model_comparison`
- `/report`
- `/prediction`
- `/model_diagnostics`

The temporary server was stopped after the smoke check.

For the final release readiness audit, Streamlit was started on port `8520` and `/` returned:

```text
HTTP/1.1 200 OK
```

In this sandboxed environment, local port binding and local HTTP checks required escalated execution. This is a sandbox permission detail, not an app startup failure.

For the v1.0 release candidate audit, Streamlit was started on port `8521` and `/` returned:

```text
HTTP/1.1 200 OK
```

Recent in-app browser checks during feature work covered:

- `/statistical_models`
- `/machine_learning`
- `/model_comparison`
- `/prediction`

No fatal page-level errors were present after the latest fixes. A full manual workflow should still be repeated after major UI changes.

## Current Test Coverage

Existing tests cover:

- variable type detection
- summary table generation
- target-aware EDA profiles, recommendations, relationship tables, and plots
- correlation matrix and relationship explorer helpers
- EDA-level outlier detection helpers and plots
- report context building plus Markdown and HTML export
- rule-based model interpretation
- LaTeX formula builders
- missing-value cleaning functions
- transformation functions
- transformation suggestion rules
- preprocessing pipeline construction
- reusable regression and classification metrics
- statistical OLS linear regression
- statistical binary logistic regression
- statistical multinomial logistic regression
- statistical ordinal logistic regression
- statistical count regression
- ML regression baselines
- ML binary classification baselines
- ML multiclass classification baselines
- ML hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- PCA / dimension reduction helpers and plots
- clustering helpers and plots
- anomaly detection helpers and plots
- model diagnostics helpers for overfitting, multicollinearity, learning curves, and OLS influence diagnostics
- ML cross-validation metrics and small-dataset handling
- ML coefficient extraction for applicable linear/logistic models
- ML feature importance and permutation importance
- decision tree explanations
- PDP/ICE helpers
- calibration helpers
- ModelRun creation/session helpers
- model artifact registry helpers
- model comparison table helpers
- prediction service
- statsmodels prediction intervals
- ML bootstrap uncertainty intervals

## Recently Updated Documentation

Updated to reflect the current implemented state:

- `docs/CURRENT_PROGRESS.md`
- `docs/PROJECT_HANDOFF.md`
- `docs/ARCHITECTURE_OVERVIEW.md`
- `docs/VALIDATION_STATUS.md`
- `docs/NEXT_TASK_ML_BASELINE_SPEC.md`

## Notes

- Tests pass with `python3 -m pytest`.
- Feature-stage development is currently paused; future sessions should treat the repository as a stabilization handoff unless the user explicitly resumes feature work.
- Saved model runs and fitted model artifacts live in Streamlit session state and are not persisted to disk.
- The repository directory is not currently a git repository.
- Generated local cache files are present, including `__pycache__` and `.pytest_cache`.

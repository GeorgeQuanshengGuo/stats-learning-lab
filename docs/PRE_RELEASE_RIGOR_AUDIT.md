# Pre-Release Rigor Audit

Last updated: 2026-06-13

This audit checks whether Stats Learning Lab is ready for public sharing as a learning-oriented Streamlit app. It focuses on safety, reproducibility, statistical interpretation boundaries, sample data, and deployment hygiene. It does not claim that the app is publication-grade or suitable for regulated production analytics.

## Audit Summary

| Area | Status | Result |
|---|---:|---|
| Automated test suite | pass | `384 passed, 1 warning in 6.42s` |
| Streamlit smoke test | pass | `/`, `/upload_data`, `/analysis_plan`, and `/report` returned HTTP 200 locally |
| Git tracked state before audit docs | pass | Working tree was clean before adding these audit documents |
| Ignored local artifacts | pass with note | Cache files, `.DS_Store`, and `outputs/workspaces/` are ignored by `.gitignore` |
| API keys / secrets scan | pass | No tracked API keys, private keys, access tokens, or Streamlit secrets were found |
| Private path / real dataset scan | pass | No tracked `/Users/`, `Downloads`, `Life Expectancy Data`, or private username strings were found |
| Large file scan | pass | No tracked files larger than 5 MB were found |
| Sample data | pass | `sample_data/` contains small synthetic CSV files documented in `docs/SAMPLE_DATASETS.md` |
| Public deployment risk | pass with caveat | Workspace snapshots are disabled by default unless `STATS_LAB_ENABLE_WORKSPACE_SNAPSHOTS=true` |
| Statistical language | pass with caveat | Reports include conservative limitations; advanced assumptions remain advisory |
| Canva visual direction | pass | A Canva visual direction draft was generated using only abstract/synthetic content |

## Commands Run

Automated tests:

```bash
python3 -m pytest
```

Result:

```text
384 passed, 1 warning in 6.42s
```

The warning is the known non-fatal `joblib/loky` physical-core detection warning during clustering tests.

Streamlit smoke test:

```bash
streamlit run app.py --server.headless true --server.port 8528
```

Checked routes:

```text
/
/upload_data
/analysis_plan
/report
```

Result:

```text
HTTP/1.1 200 OK
```

Security and release scans:

```bash
git status --short --ignored
git grep -n -E "(api[_-]?key|API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE KEY|BEGIN RSA|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-)" -- .
git grep -n -E "(/Users/|Downloads|Life Expectancy Data|quanshengguo|GeorgeQuanshengGuo)" -- .
find . -path ./.git -prune -o -type f -size +5M -print
```

Result:

- No tracked secrets or API keys found.
- No tracked private local paths or temporary real-dataset references found.
- No large files above 5 MB found.
- Ignored local files include caches, `.DS_Store`, and workspace snapshot preferences; these are not release assets.

## Sample Data Check

The documented validation datasets are present and synthetic:

- `regression_clean.csv`
- `regression_missing_outliers.csv`
- `binary_classification.csv`
- `multiclass_classification.csv`
- `ordinal_example.csv`
- `count_example.csv`
- `high_cardinality_example.csv`
- `tiny_dataset.csv`
- `edge_cases.csv`

Additional older sample files are also present and synthetic:

- `regression_sample.csv`
- `binary_classification_sample.csv`
- `multiclass_sample.csv`
- `eda_missing_values_sample.csv`

The sample data should remain small and inspectable. Do not add real user data, course datasets with licensing uncertainty, downloaded public-health files, or personal files to `sample_data/`.

## Core Workflow Validation

A lightweight programmatic workflow check was run with the synthetic sample datasets. It verified:

- `Analysis Plan` objects can be created.
- `build_data_readiness()` returns dataset and target readiness facts.
- `build_model_readiness()` returns advisory warnings for selected targets/features.
- `build_summary_table()` and `build_missing_table()` work on `regression_clean.csv`.
- OLS linear regression runs on `regression_clean.csv`.
- ML Linear Regression runs on `regression_clean.csv`.
- Statistical binary Logit runs on `binary_classification.csv`.
- ML Logistic Regression runs on `binary_classification.csv`.
- `edge_cases.csv` flags constant and ID-like columns.
- `build_report_context()` includes the analysis plan and reproducibility manifest.

One bare-Python validation command printed Streamlit context warnings because some helpers import session-aware modules outside `streamlit run`. Those warnings are expected in bare script mode and are not app failures.

## Statistical Rigor Findings

Current strengths:

- The app now includes an Analysis Plan page before EDA/modeling.
- `analysis_decision_log` records plan changes.
- Data readiness checks flag common risks such as ID-like columns, high-cardinality columns, constant columns, missing target values, and datetime columns.
- Model readiness checks flag low events-per-variable, rare classes, predictors relative to rows, ID-like predictors, possible leakage names, count-target issues, and datetime/random split risks.
- Reports include Analysis Plan, Decision Log, Rigor Checklist, Data Readiness Summary, Rigor Warnings, and Reproducibility Manifest.
- Report limitations explicitly avoid unsupported causal claims.

Remaining non-blocking risks:

- Readiness checks are advisory and cannot guarantee that the user's study design is appropriate.
- OLS, Logit, count, ordinal, and multinomial assumptions are partly checked through warnings and diagnostics, not fully formalized.
- Exploratory p-values can still be misused by users if they repeatedly try models and interpret results as confirmatory.
- Runtime-heavy features can still be slow on large or high-cardinality datasets.
- Public deployment relies on Streamlit session behavior; uploaded data should still be treated as sensitive unless the user controls the deployment and data policy.

## Public Deployment Notes

- README tells users to use synthetic or non-sensitive data only in the public demo.
- `.streamlit/secrets.toml` is ignored and no secrets file was found in the tracked files.
- Workspace snapshots use pickle and are disabled by default unless `STATS_LAB_ENABLE_WORKSPACE_SNAPSHOTS=true`.
- Local ignored file `outputs/workspaces/autosave_enabled.pkl` may exist on a developer machine, but it is ignored and should not be committed.

## Recommended Improvements Before Wider Sharing

1. Repeat the manual QA checklist with the public Streamlit deployment, not only local routes.
2. Add a visible public-demo warning near upload if it is not already prominent enough.
3. Keep workspace snapshot controls hidden on public deployments.
4. Add screenshots to README after the UI stabilizes.
5. Continue improving assumption explanations, especially for OLS, logistic regression, count models, and ordinal regression.
6. Avoid adding new models until the existing workflows have been manually validated with all synthetic datasets.

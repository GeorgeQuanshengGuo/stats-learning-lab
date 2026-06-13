# Release Checklist

Last updated: 2026-06-13

This checklist captures the final local-release readiness audit for Stats Learning Lab.

## Release Scope

Target release type: stable local release for trusted desktop use.

This is not a hosted multi-user production release. The app remains a local analysis and learning tool that requires statistical judgment.

## Audit Summary

| Area | Status | Notes |
|---|---:|---|
| Automated tests | pass | `python3 -m pytest` passes with 384 tests and 1 non-fatal joblib/loky warning. |
| Streamlit launch | pass | App starts with `streamlit run app.py --server.headless true --server.port 8528`; key routes return HTTP 200. |
| Requirements | pass with update | Dependencies now use tested major-version ranges instead of fully unpinned names. |
| Sample data | pass | `sample_data/` exists and is documented in `docs/SAMPLE_DATASETS.md`. |
| README | pass | README includes description, features, workflow, install/run instructions, sample datasets, validation status, limitations, screenshot placeholders, and disclaimer. |
| User docs | pass | User guide, quick start, troubleshooting guide, demo script, QA checklist, limitations, audits, and feature inventory exist. |
| Generated files | pass with ignore rule | Cache files and local workspace snapshots are ignored by `.gitignore`; existing local cache files are not release assets. |
| Large unnecessary files | pass | No files larger than 1 MB were found outside ignored virtual/cache paths. |
| Outputs and reports | pass with ignore rule | Generated output folders are ignored except `.gitkeep` placeholders. |
| Model artifacts | pass with ignore rule | Pickle/joblib artifacts and workspace snapshots are ignored. |
| Clean clone readiness | pass with caveats | Core files, docs, config, sample data, and requirements are present. A fresh install should use the bounded version ranges in `requirements.txt`. |

## Validation Commands Run

### Automated Tests

Command:

```bash
python3 -m pytest
```

Result:

```text
384 passed, 1 warning in 6.42s
```

Warning:

- `joblib/loky` could not detect physical CPU cores during a clustering test and fell back to logical cores.
- This warning does not indicate a failed test.

Note:

- The local environment has historically used `python3` rather than `python`.

### Streamlit Smoke Test

Command:

```bash
streamlit run app.py --server.headless true --server.port 8528
```

Smoke check:

```bash
curl -I http://localhost:8528/
```

Result:

```text
HTTP/1.1 200 OK
```

The temporary Streamlit process was stopped after the smoke test.

## Dependency Readiness

`requirements.txt` now uses version ranges based on the tested environment:

```text
streamlit>=1.57,<2
pandas>=3.0,<4
numpy>=2.4,<3
plotly>=6.7,<7
openpyxl>=3.1,<4
xlrd>=2.0,<3
scipy>=1.17,<2
statsmodels>=0.14,<0.15
scikit-learn>=1.8,<2
pytest>=9,<10
```

For a more formal external release, create a lock file from a clean virtual environment. For this local release, bounded dependency ranges are sufficient and safer than fully unpinned packages.

## Sample Data Readiness

Documented datasets:

- `sample_data/regression_clean.csv`
- `sample_data/regression_missing_outliers.csv`
- `sample_data/binary_classification.csv`
- `sample_data/multiclass_classification.csv`
- `sample_data/ordinal_example.csv`
- `sample_data/count_example.csv`
- `sample_data/high_cardinality_example.csv`
- `sample_data/tiny_dataset.csv`
- `sample_data/edge_cases.csv`

Additional older sample files are present:

- `sample_data/regression_sample.csv`
- `sample_data/binary_classification_sample.csv`
- `sample_data/multiclass_sample.csv`
- `sample_data/eda_missing_values_sample.csv`

These are small synthetic files and are not release blockers.

## Documentation Readiness

User-facing docs:

- `README.md`
- `docs/USER_GUIDE.md`
- `docs/QUICK_START.md`
- `docs/TROUBLESHOOTING.md`
- `docs/DEMO_SCRIPT.md`

Validation and stabilization docs:

- `docs/FEATURE_INVENTORY.md`
- `docs/STABILIZATION_PLAN.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/E2E_QA_CHECKLIST.md`
- `docs/SAMPLE_DATASETS.md`
- `docs/STATISTICAL_VALIDATION.md`
- `docs/ML_LEAKAGE_AUDIT.md`
- `docs/STATE_MANAGEMENT_AUDIT.md`
- `docs/EDGE_CASE_TESTING.md`
- `docs/REPORT_AUDIT.md`

## Git Ignore Readiness

`.gitignore` now excludes:

- Python caches
- pytest cache
- virtual environments
- OS/editor files
- Streamlit secrets
- generated output files
- workspace snapshots
- pickle/joblib fitted artifacts
- generated reports such as HTML/PDF/DOCX

The `outputs/figures`, `outputs/tables`, and `outputs/reports` folders keep their `.gitkeep` placeholders.

## Release Blockers

No release-blocking issues were found for a stable local release.

## Remaining Non-Blocking Risks

- Manual end-to-end QA should still be repeated with the sample datasets before sharing broadly.
- Large datasets and runtime-heavy tools can be slow.
- Local ignored files such as caches, `.DS_Store`, and workspace snapshot preferences may exist on a developer machine, but they are not release assets.
- Workspace snapshots use pickle and should remain trusted-local only.
- PDF and Word export are not implemented.
- Some docs that record older validation snapshots may show earlier test counts; this release checklist records the latest audit result.

## Recommended Version Tag

Recommended release candidate tag:

```text
v1.0.0
```

Rationale:

- The main construction phase is complete.
- Broad automated validation passes.
- The app is suitable for local v1.0 release-candidate review.
- The project still has known limitations and is not positioned as a production-hosted release.

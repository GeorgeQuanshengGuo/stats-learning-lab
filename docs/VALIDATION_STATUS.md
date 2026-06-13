# Validation Status

Last updated: 2026-06-13

This file records the latest validation pass for the local Stats Learning Lab repository.

## Latest Automated Test Run

Command:

```bash
python3 -m pytest
```

Result:

```text
384 passed, 1 warning in 6.80s
```

The warning is a non-fatal joblib/loky CPU-core detection warning that appears during machine-learning tests. It does not currently fail the suite.

## Streamlit Smoke Test

Command:

```bash
streamlit run app.py --server.headless true --server.port 8527
```

Result:

```text
Passed. The app started locally and returned HTTP 200 for:
- /
- /analysis_plan
- /report
```

The first sandboxed launch attempt failed because the local environment blocked port binding. The app launched successfully after running with local execution permission.

## Most Recent Functional Area Validated

The latest validation covered the statistical rigor workflow:

- Analysis Plan page
- `analysis_plan` session state
- `analysis_decision_log` entries
- data readiness checks
- model readiness checks
- rigor checklist
- report reproducibility manifest
- report export sections for plan, decision log, checklist, readiness, warnings, and manifest

## Current Release Risk Notes

- The rigor checks are advisory. They help users notice risks but do not guarantee a publication-ready or causal analysis.
- Some advanced statistical assumptions are still warnings rather than formal tests.
- Large datasets may still make tuning, permutation importance, bootstrap intervals, learning curves, PDP/ICE, and some diagnostics slow.
- Reports remain Markdown/HTML only.

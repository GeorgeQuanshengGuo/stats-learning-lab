# AGENTS.md

## Project
This is a Streamlit-based learning app for practicing applied statistics and machine learning workflows.

## Core principles
- Keep the original uploaded dataset immutable.
- All user cleaning operations must apply only to the working dataset.
- Every cleaning operation must produce a human-readable log entry.
- Do not build black-box AutoML. The user must choose the target, features, model, and preprocessing options.
- Use statsmodels for statistical inference models when p-values, confidence intervals, AIC, BIC, or R-style summaries are needed.
- Use scikit-learn Pipelines for predictive machine learning models.
- Avoid data leakage. Imputation, scaling, encoding, and feature selection used for modeling must be fit on the training data only.
- Keep code simple and readable for a beginner.
- Prefer small functions with clear names.
- Add or update tests when creating core logic.
- Do not implement future roadmap features unless explicitly asked.

## Tech stack
- Python
- Streamlit
- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- matplotlib
- plotly
- openpyxl

## Project structure
- app.py is the Streamlit entry point.
- pages/ contains Streamlit pages.
- src/ contains reusable logic.
- tests/ contains unit tests.
- docs/ contains handoff and architecture notes for future Codex sessions.

## Handoff docs
Future sessions should read these before major changes:
- docs/CURRENT_PROGRESS.md
- docs/PROJECT_HANDOFF.md
- docs/ARCHITECTURE_OVERVIEW.md
- docs/VALIDATION_STATUS.md
- docs/NEXT_TASK_ML_BASELINE_SPEC.md, which now records the implemented ML baseline status and recommended next ML work.

## Validation commands
Run these after changes when applicable:
```bash
python -m pytest
streamlit run app.py
```

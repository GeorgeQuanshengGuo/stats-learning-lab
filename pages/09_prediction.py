"""Streamlit page for manual predictions from saved fitted models."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.model_artifacts import initialize_model_artifacts
from src.core.model_run import get_model_runs
from src.modeling.machine_learning.regression import _build_regression_pipeline
from src.prediction.input_builder import build_input_specs
from src.prediction.ml_uncertainty import bootstrap_prediction_interval
from src.prediction.prediction_service import (
    PredictionError,
    build_prediction_log_entry,
    is_artifact_usable_for_prediction,
    predict_from_model_run,
)
from src.ui.components import render_empty_state
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


def _run_label(model_run: dict) -> str:
    """Return a readable label for selecting saved model runs."""
    model_name = model_run.get("model_name") or "Saved model"
    model_family = model_run.get("model_family") or "model"
    task_type = str(model_run.get("task_type") or "task").replace("_", " ")
    target = model_run.get("target") or "unknown target"
    features = list(model_run.get("features") or [])
    feature_summary = _format_feature_summary(features)
    timestamp = str(model_run.get("timestamp") or "")[:16].replace("T", " ")
    saved_text = f" | saved {timestamp}" if timestamp else ""
    return f"{model_name} ({model_family}) | {task_type} | target: {target} | {feature_summary}{saved_text}"


def _format_feature_summary(features: list[str]) -> str:
    """Return a compact feature summary for UI labels."""
    if not features:
        return "no listed features"
    if len(features) <= 3:
        return "features: " + ", ".join(str(feature) for feature in features)
    first_features = ", ".join(str(feature) for feature in features[:3])
    return f"features: {first_features} + {len(features) - 3} more"


def _get_artifact_value(model_run: dict, artifact: dict, key: str):
    """Prefer ModelRun metadata, then artifact metadata."""
    value = model_run.get(key)
    if not _is_empty_metadata_value(value):
        return value
    return artifact.get(key)


def _is_empty_metadata_value(value) -> bool:
    """Return True for empty metadata values without comparing DataFrames."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    if isinstance(value, pd.DataFrame):
        return value.empty
    return False


def _render_formula_block(formula_latex) -> None:
    """Render saved formulas when available."""
    if not formula_latex:
        st.info("No saved mathematical formula is available for this model run.")
        return

    st.subheader("Model Formula")
    if isinstance(formula_latex, str):
        st.latex(formula_latex)
        return

    if isinstance(formula_latex, dict):
        for label, formula in formula_latex.items():
            if not formula:
                continue
            readable_label = str(label).replace("_", " ").title()
            st.caption(readable_label)
            if str(label).lower() in {"rules", "rule_summary", "text"}:
                st.code(str(formula))
            else:
                st.latex(str(formula))
        return

    st.write(formula_latex)


def _render_coefficient_table(coefficient_table) -> None:
    """Render a coefficient table if one was saved."""
    if coefficient_table is None:
        return
    if isinstance(coefficient_table, (str, list, dict)) and not coefficient_table:
        return
    st.subheader("Coefficient Table")
    if isinstance(coefficient_table, pd.DataFrame):
        table = coefficient_table
    elif isinstance(coefficient_table, list):
        table = pd.DataFrame(coefficient_table)
    elif isinstance(coefficient_table, dict):
        table = pd.DataFrame([coefficient_table])
    else:
        st.write(coefficient_table)
        return
    if table.empty:
        return
    st.dataframe(table, use_container_width=True)


def _render_selected_model_summary(model_run: dict, artifact: dict) -> None:
    """Show readable metadata for the selected prediction model."""
    st.subheader("Selected Model")
    summary_rows = [
        ("Model", model_run.get("model_name")),
        ("Family", model_run.get("model_family")),
        ("Task", str(model_run.get("task_type") or "").replace("_", " ")),
        ("Target", model_run.get("target")),
        ("Features", ", ".join(str(feature) for feature in model_run.get("features", []))),
        ("Preprocessing", model_run.get("preprocessing") or artifact.get("preprocessing_summary")),
    ]
    summary_df = pd.DataFrame(
        [{"field": field, "value": "" if value is None else str(value)} for field, value in summary_rows]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    formula_latex = _get_artifact_value(model_run, artifact, "formula_latex")
    _render_formula_block(formula_latex)
    _render_coefficient_table(_get_artifact_value(model_run, artifact, "coefficient_table"))

    with st.expander("Technical metadata"):
        st.json(
            {
                "run_id": model_run.get("run_id"),
                "split_config": model_run.get("split_config"),
                "preprocessing": model_run.get("preprocessing"),
                "prediction_supported": artifact.get("prediction_supported"),
                "interval_supported": artifact.get("interval_supported"),
                "interval_method": artifact.get("interval_method"),
            }
        )

def _render_feature_input(spec: dict):
    """Render one Streamlit input widget from a feature spec."""
    name = spec["name"]
    kind = spec["kind"]
    default = spec.get("default")
    options = spec.get("options") or []

    if kind == "numeric":
        value = 0.0 if default in (None, "") else float(default)
        return st.number_input(
            name,
            value=value,
            key=f"prediction_input_{name}",
            help="Enter the raw feature value for this single prediction.",
        )
    if kind == "boolean":
        return st.checkbox(
            name,
            value=bool(default),
            key=f"prediction_input_{name}",
            help="Choose True or False for this feature.",
        )
    if options:
        index = options.index(default) if default in options else 0
        return st.selectbox(
            name,
            options,
            index=index,
            key=f"prediction_input_{name}",
            help="Choose the category value for this feature.",
        )
    return st.text_input(
        name,
        value="" if default is None else str(default),
        key=f"prediction_input_{name}",
        help="Enter the raw feature value for this single prediction.",
    )


def _interval_figure(interval_rows: list[dict]) -> go.Figure | None:
    """Build a compact interval chart from prediction interval rows."""
    if not interval_rows:
        return None
    row = interval_rows[0]
    point = row.get("point_prediction") or row.get("predicted_mean") or row.get("predicted_expected_count")
    lower = row.get("lower_bound") or row.get("obs_ci_lower") or row.get("mean_ci_lower")
    upper = row.get("upper_bound") or row.get("obs_ci_upper") or row.get("mean_ci_upper")
    if point is None or lower is None or upper is None:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[float(lower), float(upper)],
            y=["Prediction", "Prediction"],
            mode="lines",
            name="Interval",
            line={"width": 8, "color": "#93c5fd"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[float(point)],
            y=["Prediction"],
            mode="markers",
            name="Point prediction",
            marker={"size": 14, "color": "#2563eb"},
        )
    )
    figure.update_layout(
        title="Prediction interval",
        xaxis_title="Predicted value",
        yaxis_title="",
        showlegend=True,
    )
    return figure


def _probability_figure(class_probabilities: dict) -> go.Figure | None:
    """Build a probability bar chart from class probability output."""
    if not class_probabilities:
        return None
    classes = list(class_probabilities.keys())
    probabilities = [float(value) for value in class_probabilities.values()]
    figure = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probabilities,
                marker_color="#2563eb",
                hovertemplate="Class=%{x}<br>Probability=%{y:.3f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Predicted class probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
    )
    return figure


def _display_prediction_result(result: dict) -> None:
    """Display a prediction result dictionary."""
    task_type = result.get("task_type")
    if task_type == "regression":
        st.metric("Predicted value", result["predicted_value"])
        if result.get("interval_available"):
            st.subheader("Prediction Interval")
            st.dataframe(pd.DataFrame(result["interval"]), use_container_width=True)
            interval_figure = _interval_figure(result.get("interval"))
            if interval_figure is not None:
                render_plotly_chart_card(
                    interval_figure,
                    title="Regression prediction interval",
                    chart_type="prediction_interval_plot",
                    source_page="Prediction",
                    variables_used=[],
                    method="Saved model prediction interval or empirical uncertainty interval",
                    sample_size=1,
                    missing_handling="Uses the single user-entered observation.",
                    next_step="Report the interval method clearly and avoid treating it as certainty.",
                )
        else:
            st.info(result.get("interval_message", "Interval is not available for this model."))
        return

    if task_type == "binary_classification":
        st.metric("Predicted class", result["predicted_class"])
        if result.get("positive_class_probability") is not None:
            st.metric(
                f"Probability for positive class ({result.get('positive_class')})",
                f"{result['positive_class_probability']:.4f}",
            )
        st.write(f"Selected threshold: `{result.get('threshold')}`")
        if result.get("class_probabilities"):
            st.subheader("Class Probabilities")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"class": class_name, "probability": probability}
                        for class_name, probability in result["class_probabilities"].items()
                    ]
                ),
                use_container_width=True,
            )
            probability_figure = _probability_figure(result["class_probabilities"])
            if probability_figure is not None:
                render_plotly_chart_card(
                    probability_figure,
                    title="Predicted class probabilities",
                    chart_type="probability_distribution",
                    source_page="Prediction",
                    variables_used=[],
                    method="Single-row predicted probabilities",
                    sample_size=1,
                    missing_handling="Uses the single user-entered observation.",
                    next_step="Review threshold and calibration before using probabilities for decisions.",
                )
        if result.get("interval_available"):
            st.subheader("Probability Confidence Interval")
            st.dataframe(pd.DataFrame(result["interval"]), use_container_width=True)
        elif result.get("interval_message"):
            st.info(result["interval_message"])
        return

    st.metric("Predicted class", result.get("predicted_class"))
    if result.get("class_probabilities"):
        st.subheader("Class Probabilities")
        st.dataframe(pd.DataFrame(result["class_probabilities"].items(), columns=["class", "probability"]))
        probability_figure = _probability_figure(result["class_probabilities"])
        if probability_figure is not None:
            render_plotly_chart_card(
                probability_figure,
                title="Predicted class probabilities",
                chart_type="probability_distribution",
                source_page="Prediction",
                variables_used=[],
                method="Single-row predicted probabilities",
                sample_size=1,
                missing_handling="Uses the single user-entered observation.",
                next_step="Use probabilities as model output, not as certainty.",
            )


def _build_ml_regression_pipeline_factory(model_run: dict, artifact: dict):
    """Create fresh ML regression Pipelines that match a saved model run."""
    preprocessing = model_run.get("preprocessing") or artifact.get("preprocessing_summary") or {}
    split_config = model_run.get("split_config") or {}
    tree_settings = preprocessing.get("decision_tree_settings") or {}
    model_name = model_run.get("model_name")
    features = list(artifact.get("features") or model_run.get("features") or [])
    random_state = int(split_config.get("random_state", 42))
    scale_numeric = bool(preprocessing.get("scale_numeric", False))

    def factory(training_df: pd.DataFrame):
        return _build_regression_pipeline(
            model_name=model_name,
            train_df=training_df,
            feature_columns=features,
            random_state=random_state,
            scale_numeric=scale_numeric,
            tree_max_depth=tree_settings.get("max_depth"),
            tree_min_samples_leaf=int(tree_settings.get("min_samples_leaf", 1)),
            tree_min_samples_split=int(tree_settings.get("min_samples_split", 2)),
        )

    factory.fitted_pipeline = artifact.get("fitted_pipeline")
    return factory


def _is_ml_regression_run(model_run: dict) -> bool:
    """Return True for saved machine learning regression runs."""
    return (
        model_run.get("task_type") == "regression"
        and model_run.get("model_family") == "machine_learning"
    )


def _is_ml_binary_classification_run(model_run: dict) -> bool:
    """Return True for saved machine learning binary classification runs."""
    return (
        model_run.get("task_type") == "binary_classification"
        and model_run.get("model_family") == "machine_learning"
    )


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Prediction",
    purpose="Enter raw feature values and generate predictions from saved fitted model artifacts.",
    when_to_use="Use this page after fitting and saving a model run with a usable fitted artifact.",
    common_mistake="Do not treat predicted probabilities or intervals as certainty; review model assumptions and calibration.",
    key_terms=["prediction_interval", "prediction_probability"],
    next_step="Log useful predictions and include them in the report if needed.",
    next_page="pages/08_report.py",
    tags=["Prediction", "Saved models"],
)

model_runs = get_model_runs()
model_artifacts = initialize_model_artifacts()
st.session_state.setdefault("prediction_log", [])

if not model_runs:
    render_empty_state(
        "No saved model runs yet",
        "Fit and save a model before using the prediction console.",
        action_label="Go to Machine Learning",
        action_page="pages/07_machine_learning.py",
        icon="Prediction:",
    )
else:
    usable_runs = [
        run
        for run in model_runs
        if is_artifact_usable_for_prediction(run, model_artifacts.get(run.get("run_id")))
    ]
    missing_artifact_runs = [
        run
        for run in model_runs
        if not is_artifact_usable_for_prediction(run, model_artifacts.get(run.get("run_id")))
    ]

    if missing_artifact_runs:
        st.warning(
            f"{len(missing_artifact_runs)} saved model run(s) do not have usable fitted artifacts in this session."
        )

    if not usable_runs:
        st.info("No saved model runs with usable fitted artifacts are available for prediction.")
    else:
        selected_index = st.selectbox(
            "Saved model run",
            list(range(len(usable_runs))),
            format_func=lambda index: _run_label(usable_runs[index]),
            key="prediction_saved_model_run",
            help="Choose a saved run with a fitted model artifact available in this session.",
        )
        selected_run = usable_runs[selected_index]
        selected_artifact = model_artifacts[selected_run["run_id"]]

        _render_selected_model_summary(selected_run, selected_artifact)

        st.subheader("Input Values")
        working_df = st.session_state.get("working_df")
        features = list(selected_artifact.get("features") or selected_run.get("features") or [])
        specs = build_input_specs(features, working_df)
        input_values = {spec["name"]: _render_feature_input(spec) for spec in specs}

        threshold = None
        if selected_run.get("task_type") == "binary_classification":
            default_threshold = float((selected_run.get("preprocessing") or {}).get("threshold", 0.5))
            threshold = st.slider(
                "Classification threshold",
                min_value=0.05,
                max_value=0.95,
                value=default_threshold,
                step=0.01,
                help="Probability cutoff used to turn predicted probability into a predicted class.",
            )

        bootstrap_enabled = False
        bootstrap_n = 50
        if _is_ml_regression_run(selected_run):
            st.subheader("Empirical Uncertainty")
            bootstrap_enabled = st.checkbox(
                "Compute bootstrap empirical prediction interval",
                value=False,
                help=(
                    "This refits the full preprocessing + model Pipeline on resampled data. "
                    "It can be slow for larger datasets."
                ),
            )
            if bootstrap_enabled:
                st.warning(
                    "Bootstrap intervals are empirical uncertainty intervals, not classical "
                    "statistical confidence intervals."
                )
                bootstrap_n = st.number_input(
                    "Bootstrap repetitions",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of bootstrap refits used to estimate the empirical interval.",
                )

        if _is_ml_binary_classification_run(selected_run):
            st.info(
                "Probability uncertainty intervals for ML classification are not available yet. "
                "They require calibration, bootstrap, or conformal methods."
            )

        if st.button(
            "Predict",
            help="Generate a prediction for the entered feature values without modifying the dataset.",
        ):
            try:
                result = predict_from_model_run(
                    selected_run,
                    selected_artifact,
                    input_values,
                    threshold=threshold,
                )
                if bootstrap_enabled:
                    if working_df is None:
                        raise PredictionError("A working dataset is needed to compute bootstrap intervals.")
                    interval = bootstrap_prediction_interval(
                        base_pipeline_factory=_build_ml_regression_pipeline_factory(
                            selected_run,
                            selected_artifact,
                        ),
                        df=working_df,
                        target_column=selected_run["target"],
                        feature_columns=features,
                        input_row=input_values,
                        n_bootstrap=int(bootstrap_n),
                        alpha=0.05,
                        random_state=int((selected_run.get("split_config") or {}).get("random_state", 42)),
                    )
                    interval["point_prediction"] = result["predicted_value"]
                    result["interval"] = [interval]
                    result["interval_available"] = True
                    result["interval_message"] = interval["interval_explanation"]
                elif _is_ml_binary_classification_run(selected_run):
                    result["interval_message"] = (
                        "Probability uncertainty requires calibration, bootstrap, or conformal "
                        "methods and is not available for ML classification yet."
                    )
            except PredictionError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not generate prediction: {error}")
            else:
                _display_prediction_result(result)
                log_entry = build_prediction_log_entry(
                    run_id=selected_run["run_id"],
                    model_name=selected_run["model_name"],
                    input_values=input_values,
                    prediction_result=result,
                )
                st.session_state["prediction_log"].append(log_entry)

    if st.session_state.get("prediction_log"):
        st.subheader("Prediction Log")
        st.dataframe(pd.DataFrame(st.session_state["prediction_log"]), use_container_width=True)

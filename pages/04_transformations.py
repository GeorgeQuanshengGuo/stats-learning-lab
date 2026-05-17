"""Streamlit page for transformations and feature engineering."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src.core.state import apply_transformation_result
from src.data.transformation_suggestions import build_transformation_suggestions
from src.data.transformations import (
    apply_boxcox_transform,
    apply_cube_root_transform,
    apply_log_transform,
    apply_log1p_transform,
    apply_reciprocal_transform,
    apply_sqrt_transform,
    apply_square_transform,
    apply_yeojohnson_transform,
    create_interaction_term,
    create_polynomial_feature,
    create_ratio_feature,
    inverse_transform_available,
)
from src.ui.page_templates import render_dataset_required_empty_state, render_educational_page_header
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles
from src.visualization.transformation_plots import plot_before_after_histograms


SINGLE_COLUMN_METHODS = {
    "Log": apply_log_transform,
    "Log1p": apply_log1p_transform,
    "Square root": apply_sqrt_transform,
    "Cube root": apply_cube_root_transform,
    "Reciprocal": apply_reciprocal_transform,
    "Square": apply_square_transform,
    "Box-Cox": apply_boxcox_transform,
    "Yeo-Johnson": apply_yeojohnson_transform,
}

TARGET_METHODS = {
    "log": apply_log_transform,
    "log1p": apply_log1p_transform,
    "sqrt": apply_sqrt_transform,
    "boxcox": apply_boxcox_transform,
    "yeojohnson": apply_yeojohnson_transform,
}


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that can be treated as numeric."""
    return df.select_dtypes(include="number").columns.tolist()


def _preview_stats(df: pd.DataFrame, source_column: str, new_column: str) -> pd.DataFrame:
    """Build before/after summary statistics for preview."""
    return pd.DataFrame(
        {
            "source": df[source_column].describe(),
            "transformed": df[new_column].describe(),
        }
    )


def _run_selected_transformation(
    df: pd.DataFrame,
    method: str,
    column: str,
    new_column: str,
    second_column: str | None,
    power: int,
):
    """Run the selected transformation for preview."""
    if method in SINGLE_COLUMN_METHODS:
        return SINGLE_COLUMN_METHODS[method](df, column, new_column)
    if method == "Interaction term":
        return create_interaction_term(df, column, second_column, new_column)
    if method == "Ratio feature":
        return create_ratio_feature(df, column, second_column, new_column)
    return create_polynomial_feature(df, column, power, new_column)


def _default_new_column_name(method: str, source: str, second_column: str | None, power: int) -> str:
    """Create a readable default column name for the selected method."""
    suffix_by_method = {
        "Log": "log",
        "Log1p": "log1p",
        "Square root": "sqrt",
        "Cube root": "cuberoot",
        "Reciprocal": "reciprocal",
        "Square": "squared",
        "Box-Cox": "boxcox",
        "Yeo-Johnson": "yeojohnson",
        "Polynomial feature": f"power_{power}",
    }
    if method == "Interaction term":
        return f"{source}_x_{second_column}"
    if method == "Ratio feature":
        return f"{source}_over_{second_column}"
    return f"{source}_{suffix_by_method[method]}"


def _store_transformation_preview(
    preview_df: pd.DataFrame,
    log_entry: dict,
    source_column: str,
    context: str,
) -> None:
    """Store preview data in session state."""
    st.session_state["transformation_preview_df"] = preview_df
    st.session_state["transformation_preview_log"] = log_entry
    st.session_state["transformation_preview_source"] = source_column
    st.session_state["transformation_preview_new_column"] = log_entry["new_column"]
    st.session_state["transformation_preview_context"] = context


def _render_transformation_preview(working_df: pd.DataFrame, key_suffix: str) -> None:
    """Render the shared transformation preview and confirmation controls."""
    preview_df = st.session_state.get("transformation_preview_df")
    log_entry = st.session_state.get("transformation_preview_log")
    preview_source = st.session_state.get("transformation_preview_source")
    preview_new_column = st.session_state.get("transformation_preview_new_column")
    preview_context = st.session_state.get("transformation_preview_context")

    if preview_df is None or log_entry is None or preview_context != key_suffix:
        return

    st.subheader("Preview Summary")
    st.json(log_entry)
    st.dataframe(_preview_stats(preview_df, preview_source, preview_new_column), use_container_width=True)

    st.subheader("Preview Histogram")
    render_plotly_chart_card(
        plot_before_after_histograms(
            working_df[preview_source],
            preview_df[preview_new_column],
            preview_source,
            preview_new_column,
        ),
        title=f"Before and after: {preview_source}",
        chart_type="histogram",
        source_page="Transformations",
        variables_used=[preview_source, preview_new_column],
        method="Before/after histogram comparison",
        sample_size=int(preview_df[[preview_new_column]].dropna().shape[0]),
        missing_handling="Missing source or transformed values are not shown in the histogram.",
        next_step="Confirm the transformed variable only if the new distribution is useful for analysis.",
    )

    if st.button(
        "Confirm and create transformed variable",
        key=f"confirm_transform_{key_suffix}",
        help="Create the previewed new column in working_df. The original dataset and source column stay unchanged.",
    ):
        apply_transformation_result(preview_df, log_entry)
        st.session_state["transformation_preview_df"] = None
        st.session_state["transformation_preview_log"] = None
        st.session_state["transformation_preview_source"] = None
        st.session_state["transformation_preview_new_column"] = None
        st.session_state["transformation_preview_context"] = None
        st.success("Transformed variable created in working_df.")
        st.rerun()


def _render_transformation_log() -> None:
    """Render transformation log table."""
    st.subheader("Transformation Log")
    transformation_log = st.session_state.get("transformation_log", [])
    if transformation_log:
        st.dataframe(pd.DataFrame(transformation_log), use_container_width=True)
    else:
        st.info("No transformations have been applied yet.")


def _target_default_new_column(method: str, target: str) -> str:
    """Create target-oriented names such as log_price."""
    return f"{method}_{target}"


def _target_log_entry(log_entry: dict, original_target: str, transformed_target: str) -> dict:
    """Add target transformation metadata to a normal transformation log entry."""
    updated = log_entry.copy()
    updated["original_target"] = original_target
    updated["transformed_target"] = transformed_target
    updated["inverse_transform_available"] = inverse_transform_available(updated["method"])
    return updated


def _target_candidate_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build eligibility and before/after skewness rows for target transforms."""
    values = pd.to_numeric(df[target], errors="coerce")
    skew_before = float(values.dropna().skew()) if values.dropna().shape[0] >= 3 else np.nan
    rows = []

    for method, function in TARGET_METHODS.items():
        preview_column = f"__target_preview_{method}_{target}"
        try:
            preview_df, log_entry = function(df, target, preview_column)
        except ValueError as error:
            rows.append(
                {
                    "method": method,
                    "eligibility": "not eligible",
                    "skewness_before": skew_before,
                    "skewness_after": np.nan,
                    "lambda": np.nan,
                    "warning": str(error),
                }
            )
        else:
            transformed = pd.to_numeric(preview_df[preview_column], errors="coerce")
            skew_after = float(transformed.dropna().skew()) if transformed.dropna().shape[0] >= 3 else np.nan
            rows.append(
                {
                    "method": method,
                    "eligibility": "eligible",
                    "skewness_before": skew_before,
                    "skewness_after": skew_after,
                    "lambda": log_entry["parameters"].get("lambda"),
                    "warning": "",
                }
            )

    return pd.DataFrame(rows)


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Transformations",
    purpose="Create transformed variables and engineered features without overwriting source columns.",
    when_to_use="Use this page when EDA or modeling diagnostics suggest skewness, nonlinear relationships, or useful feature combinations.",
    common_mistake="Do not transform variables just to force a model; choose transformations that match the analysis question.",
    key_terms=[("box_cox", "Box-Cox"), ("yeo_johnson", "Yeo-Johnson"), ("interaction_term", "Interaction term")],
    next_step="After creating useful variables, choose a statistical or machine learning model.",
    next_page="pages/05_statistical_models.py",
    tags=["Feature engineering", "Data preparation"],
)

working_df = st.session_state.get("working_df")
original_df = st.session_state.get("original_df")

if working_df is None or original_df is None:
    render_dataset_required_empty_state()
else:
    numeric_columns = _numeric_columns(working_df)

    if not numeric_columns:
        st.warning("No numeric variables are available for transformations.")
    else:
        manual_tab, target_tab, suggestions_tab, log_tab = st.tabs(
            ["Manual Transform", "Target Transformation", "Transformation Suggestions", "Log"]
        )

        with manual_tab:
            method = st.selectbox(
                "Transformation method",
                [
                    "Log",
                    "Log1p",
                    "Square root",
                    "Cube root",
                    "Reciprocal",
                    "Square",
                    "Box-Cox",
                    "Yeo-Johnson",
                    "Interaction term",
                    "Ratio feature",
                    "Polynomial feature",
                ],
                help=(
                    "Choose a new variable to create. Log/Box-Cox need positive values; "
                    "Yeo-Johnson can handle zero and negative values."
                ),
            )
            source_column = st.selectbox(
                "Numeric source variable",
                numeric_columns,
                help="Choose the numeric variable to transform. The source column is not overwritten.",
            )

            second_column = None
            if method in {"Interaction term", "Ratio feature"}:
                second_options = [column for column in numeric_columns if column != source_column]
                if not second_options:
                    st.warning("This method needs at least two numeric variables.")
                    st.stop()
                second_column = st.selectbox(
                    "Second numeric variable",
                    second_options,
                    help="Choose the second variable for an interaction or ratio feature.",
                )

            power = 2
            if method == "Polynomial feature":
                power = st.number_input(
                    "Power",
                    min_value=2,
                    max_value=10,
                    value=2,
                    step=1,
                    help="Power for the polynomial feature. A square uses power 2.",
                )

            default_new_column = _default_new_column_name(method, source_column, second_column, int(power))
            new_column = st.text_input(
                "New column name",
                value=default_new_column,
                help="Name of the new transformed column. Existing source columns are not overwritten.",
            )

            if st.button(
                "Preview transformation",
                help="Preview summary statistics and charts before creating the new working-dataset column.",
            ):
                try:
                    preview_df, log_entry = _run_selected_transformation(
                        working_df,
                        method,
                        source_column,
                        new_column,
                        second_column,
                        int(power),
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    _store_transformation_preview(preview_df, log_entry, source_column, "manual")

            _render_transformation_preview(working_df, "manual")

        with target_tab:
            st.info(
                "Use target transformation mainly for continuous y. Do not transform binary, "
                "multiclass, ordinal, or count targets just to force linear regression. "
                "For binary targets, use logistic regression. For count targets, consider "
                "Poisson or Negative Binomial models."
            )
            target_column = st.selectbox(
                "Continuous target variable",
                numeric_columns,
                key="target_transform_y",
                help="Choose a continuous outcome to explore before linear regression. Do not use this for binary or count targets.",
            )
            candidate_table = _target_candidate_table(working_df, target_column)
            st.dataframe(candidate_table, use_container_width=True)

            eligible_methods = candidate_table.loc[
                candidate_table["eligibility"] == "eligible",
                "method",
            ].tolist()

            if not eligible_methods:
                st.warning("No target transformations are eligible for this variable.")
            else:
                target_method = st.selectbox(
                    "Target transformation to preview",
                    eligible_methods,
                    key="target_transform_method",
                    help="Choose an eligible target transformation. Eligibility depends on the target values.",
                )
                target_new_column = st.text_input(
                    "New target column name",
                    value=_target_default_new_column(target_method, target_column),
                    key="target_transform_new_column",
                    help="Name for the transformed target column, such as log_price.",
                )

                if st.button(
                    "Preview target transformation",
                    help="Preview the transformed target before creating a new working-dataset column.",
                ):
                    try:
                        preview_df, log_entry = TARGET_METHODS[target_method](
                            working_df,
                            target_column,
                            target_new_column,
                        )
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        log_entry["notes"] = (
                            f"Created transformed continuous target candidate {target_new_column}. "
                            "Linear regression fitting logic was not changed."
                        )
                        log_entry = _target_log_entry(log_entry, target_column, target_new_column)
                        _store_transformation_preview(preview_df, log_entry, target_column, "target")

                _render_transformation_preview(working_df, "target")

        with suggestions_tab:
            suggestions = build_transformation_suggestions(working_df)
            display_suggestions = suggestions.copy()
            display_suggestions["suggested_methods"] = display_suggestions["suggested_methods"].apply(
                lambda methods: ", ".join(methods)
            )
            st.dataframe(display_suggestions, use_container_width=True)

            actionable = suggestions[suggestions["suggested_methods"].map(bool)].reset_index(drop=True)
            if actionable.empty:
                st.info("No actionable transformation suggestions are available.")
            else:
                selected_label = st.selectbox(
                    "Suggested variable",
                    [
                        f"{row.variable} | {row.detected_issue}"
                        for row in actionable.itertuples(index=False)
                    ],
                    help="Choose a variable with an automatically detected transformation suggestion.",
                )
                selected_index = [
                    f"{row.variable} | {row.detected_issue}"
                    for row in actionable.itertuples(index=False)
                ].index(selected_label)
                selected_row = actionable.iloc[selected_index]
                suggested_method = st.selectbox(
                    "Suggested method",
                    selected_row["suggested_methods"],
                    help="Choose one suggested transformation to preview. Suggestions are not applied automatically.",
                )
                suggested_new_column = st.text_input(
                    "New suggested column name",
                    value=_default_new_column_name(suggested_method, selected_row["variable"], None, 2),
                    help="Name for the new column created from the suggested transformation.",
                )

                if st.button(
                    "Preview suggested transformation",
                    help="Preview the suggested transformation before creating a new column.",
                ):
                    try:
                        preview_df, log_entry = _run_selected_transformation(
                            working_df,
                            suggested_method,
                            selected_row["variable"],
                            suggested_new_column,
                            None,
                            2,
                        )
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        _store_transformation_preview(
                            preview_df,
                            log_entry,
                            selected_row["variable"],
                            "suggestions",
                        )

                _render_transformation_preview(working_df, "suggestions")

        with log_tab:
            _render_transformation_log()

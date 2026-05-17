"""Streamlit page for simple exploratory data analysis."""

import pandas as pd
import streamlit as st

from src.core.state import apply_transformation_result
from src.eda.correlation import (
    build_correlation_matrix,
    build_correlation_pairs_table,
    identify_high_correlation_pairs,
)
from src.eda.missing import build_missing_table, build_row_missing_summary
from src.eda.outliers import (
    OUTLIER_WARNING,
    add_outlier_flag_column,
    run_outlier_detection,
)
from src.eda.relationships import (
    binary_target_categorical_feature_summary,
    binary_target_numeric_feature_summary,
    categorical_categorical_relationship_tables,
    categorical_target_categorical_feature_tables,
    categorical_target_numeric_feature_summary,
    datetime_numeric_relationship_summary,
    numeric_categorical_relationship_summary,
    numeric_target_categorical_feature_summary,
    numeric_target_numeric_feature_summary,
    numeric_numeric_relationship_summary,
)
from src.eda.summary import build_summary_table, dataset_overview
from src.eda.target_aware import (
    build_model_recommendations,
    build_target_profile,
    is_categorical_target,
    is_numeric_target,
    variable_role,
)
from src.visualization.eda_plots import (
    plot_categorical_bar,
    plot_datetime_counts,
    plot_missing_bar,
    plot_numeric_boxplot,
    plot_numeric_histogram,
)
from src.visualization.outlier_plots import (
    plot_outlier_boxplot,
    plot_outlier_histogram,
    plot_outlier_pca,
    plot_outlier_scatter,
)
from src.visualization.correlation_plots import plot_correlation_heatmap
from src.visualization.relationship_plots import (
    plot_categorical_relationship_bar,
    plot_datetime_numeric_line,
    plot_numeric_by_category_box,
    plot_numeric_by_category_violin,
    plot_numeric_vs_numeric_scatter,
)
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.page_templates import render_dataset_required_empty_state, render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Exploratory Data Analysis",
    purpose="Inspect data quality, distributions, missingness, correlations, relationships, and outliers before modeling.",
    when_to_use="Use this page after upload and before cleaning, transformations, or model fitting.",
    common_mistake="Exploratory patterns are clues, not causal conclusions.",
    key_terms=[("summary_table", "Summary table"), "correlation", "outlier"],
    next_step="If data quality issues appear, move to Data Cleaning or Transformations.",
    next_page="pages/03_data_cleaning.py",
    tags=["EDA", "Data understanding"],
)


def _detected_type_lookup(summary, column: str) -> str:
    """Return the detected type from the summary table."""
    return summary.loc[summary["variable"] == column, "detected_type"].iloc[0]


def _explorer_role(detected_type: str, series: pd.Series) -> str:
    """Map a column to the broad relationship explorer role."""
    if detected_type == "datetime" or pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if detected_type in {"continuous_numeric", "discrete_numeric"}:
        return "numeric"
    if pd.api.types.is_numeric_dtype(series) and detected_type not in {"binary", "id_like", "constant"}:
        return "numeric"
    if detected_type in {"binary", "nominal_categorical", "ordinal_categorical_candidate", "constant"}:
        return "categorical"
    return "other"


def _outlier_thresholds(outlier_result: dict, column: str) -> tuple[float | None, float | None]:
    """Return raw lower/upper thresholds for histogram overlays when available."""
    details = outlier_result.get("details") or {}
    lower = (details.get("lower_fences") or details.get("lower_thresholds") or {}).get(column)
    upper = (details.get("upper_fences") or details.get("upper_thresholds") or {}).get(column)
    return lower, upper


def _rows_used(df: pd.DataFrame, columns: list[str]) -> int:
    """Count rows available for a chart after dropping missing values in selected columns."""
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return len(df)
    return int(df[available_columns].dropna().shape[0])


def _render_eda_chart(
    figure,
    title: str,
    chart_type: str,
    variables_used: list[str],
    method: str,
    sample_size: int | None,
    next_step: str,
    missing_handling: str = "Rows with missing values in the plotted variables are excluded.",
) -> None:
    """Render one EDA figure with consistent chart context."""
    render_plotly_chart_card(
        figure,
        title=title,
        chart_type=chart_type,
        source_page="EDA",
        variables_used=variables_used,
        method=method,
        sample_size=sample_size,
        missing_handling=missing_handling,
        next_step=next_step,
    )

working_df = st.session_state.get("working_df")
if working_df is None:
    render_dataset_required_empty_state()
else:
    st.subheader("Dataset Overview")
    st.json(dataset_overview(working_df))

    st.subheader("Column Summary")
    summary = build_summary_table(working_df)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Variable Plot")
    selected_variable = st.selectbox(
        "Select a variable",
        working_df.columns,
        help="Choose one column to inspect. The app uses the detected variable type to choose an appropriate chart.",
    )
    detected_type = summary.loc[
        summary["variable"] == selected_variable,
        "detected_type",
    ].iloc[0]

    st.caption(f"Detected type: {detected_type}")

    if detected_type == "continuous_numeric":
        _render_eda_chart(
            plot_numeric_histogram(working_df, selected_variable),
            title=f"Distribution of {selected_variable}",
            chart_type="histogram",
            variables_used=[selected_variable],
            method="Histogram",
            sample_size=_rows_used(working_df, [selected_variable]),
            missing_handling="Missing values are excluded from this univariate chart.",
            next_step="Use the boxplot and summary table to check skewness, spread, and possible outliers.",
        )
        _render_eda_chart(
            plot_numeric_boxplot(working_df, selected_variable),
            title=f"Boxplot of {selected_variable}",
            chart_type="boxplot",
            variables_used=[selected_variable],
            method="Boxplot with outlier points",
            sample_size=_rows_used(working_df, [selected_variable]),
            next_step="Investigate possible outliers before cleaning or modeling.",
        )
    elif detected_type == "discrete_numeric":
        _render_eda_chart(
            plot_categorical_bar(working_df, selected_variable),
            title=f"Value counts for {selected_variable}",
            chart_type="categorical_bar",
            variables_used=[selected_variable],
            method="Frequency bar chart",
            sample_size=_rows_used(working_df, [selected_variable]),
            next_step="Check whether low-frequency values should be reviewed before modeling.",
        )
    elif detected_type in {"binary", "nominal_categorical"}:
        _render_eda_chart(
            plot_categorical_bar(working_df, selected_variable),
            title=f"Category counts for {selected_variable}",
            chart_type="categorical_bar",
            variables_used=[selected_variable],
            method="Frequency bar chart",
            sample_size=_rows_used(working_df, [selected_variable]),
            next_step="Review rare or unexpected categories before modeling.",
        )
    elif detected_type == "datetime":
        _render_eda_chart(
            plot_datetime_counts(working_df, selected_variable),
            title=f"Observations over time for {selected_variable}",
            chart_type="datetime_counts",
            variables_used=[selected_variable],
            method="Daily observation count",
            sample_size=_rows_used(working_df, [selected_variable]),
            missing_handling="Invalid or missing datetimes are excluded from this chart.",
            next_step="Check whether date coverage or collection gaps affect the analysis.",
        )
    else:
        st.info("No basic univariate plot is available for this variable type yet.")

    st.subheader("Outlier Detection")
    st.warning(OUTLIER_WARNING)
    outlier_numeric_columns = [
        column
        for column in working_df.columns
        if _explorer_role(_detected_type_lookup(summary, column), working_df[column]) == "numeric"
    ]
    if not outlier_numeric_columns:
        st.info("No numeric variables are available for outlier detection.")
    else:
        selected_outlier_columns = st.multiselect(
            "Numeric variables for outlier detection",
            outlier_numeric_columns,
            default=outlier_numeric_columns[:1],
            key="eda_outlier_columns",
            help="Choose numeric variables used to flag unusual rows. This only identifies potential outliers; it does not delete data.",
        )
        outlier_method = st.selectbox(
            "Outlier detection method",
            [
                "IQR rule",
                "Z-score",
                "Modified Z-score",
                "Mahalanobis distance",
                "Isolation Forest",
                "Local Outlier Factor",
            ],
            key="eda_outlier_method",
            help="Choose the rule or algorithm used to flag unusual observations. A flagged row is not automatically an error.",
        )
        iqr_multiplier = 1.5
        z_threshold = 3.0
        modified_z_threshold = 3.5
        mahalanobis_quantile = 0.975
        contamination = 0.1
        n_neighbors = 20
        random_state = 42
        if outlier_method == "IQR rule":
            iqr_multiplier = st.slider(
                "IQR multiplier",
                0.5,
                5.0,
                1.5,
                0.1,
                help="Larger values flag fewer outliers. The common default is 1.5 times the interquartile range.",
            )
        elif outlier_method == "Z-score":
            z_threshold = st.slider(
                "Z-score threshold",
                1.0,
                6.0,
                3.0,
                0.1,
                help="Rows farther than this many standard deviations from the mean are flagged.",
            )
        elif outlier_method == "Modified Z-score":
            modified_z_threshold = st.slider(
                "Modified Z-score threshold",
                1.0,
                8.0,
                3.5,
                0.1,
                help="A robust outlier threshold based on the median and MAD. The common default is 3.5.",
            )
        elif outlier_method == "Mahalanobis distance":
            mahalanobis_quantile = st.slider(
                "Mahalanobis chi-square quantile",
                0.90,
                0.999,
                0.975,
                0.005,
                help="Higher values flag fewer multivariate outliers.",
            )
        elif outlier_method == "Isolation Forest":
            contamination = st.slider(
                "Contamination",
                0.01,
                0.49,
                0.10,
                0.01,
                help="Approximate fraction of rows expected to be unusual.",
            )
            random_state = st.number_input(
                "Outlier random state",
                min_value=0,
                value=42,
                step=1,
                help="Keeps randomized outlier results reproducible.",
            )
        elif outlier_method == "Local Outlier Factor":
            contamination = st.slider(
                "Contamination",
                0.01,
                0.49,
                0.10,
                0.01,
                key="lof_contamination",
                help="Approximate fraction of rows expected to be unusual.",
            )
            n_neighbors = st.number_input(
                "Number of neighbors",
                min_value=1,
                value=20,
                step=1,
                help="Number of nearby rows used to judge whether a row is locally unusual.",
            )

        if st.button(
            "Run outlier detection",
            help="Flag unusual rows using the selected method. This does not modify the working dataset.",
        ):
            try:
                st.session_state["latest_eda_outlier_result"] = run_outlier_detection(
                    working_df,
                    numeric_columns=selected_outlier_columns,
                    method=outlier_method,
                    iqr_multiplier=iqr_multiplier,
                    z_threshold=z_threshold,
                    modified_z_threshold=modified_z_threshold,
                    mahalanobis_quantile=mahalanobis_quantile,
                    contamination=contamination,
                    n_neighbors=int(n_neighbors),
                    random_state=int(random_state),
                )
            except Exception as error:
                st.error(f"Could not run outlier detection: {error}")

        outlier_result = st.session_state.get("latest_eda_outlier_result")
        if outlier_result:
            st.write("Outlier summary")
            summary_values = outlier_result["summary"]
            col_1, col_2, col_3 = st.columns(3)
            col_1.metric("Flagged rows", summary_values["flagged_count"])
            col_2.metric("Flagged percentage", f"{summary_values['flagged_percentage']:.2%}")
            col_3.metric("Rows evaluated", summary_values["rows_evaluated"])

            st.write("Score table")
            st.dataframe(outlier_result["result_table"], use_container_width=True)
            st.write("Flagged rows")
            if outlier_result["flagged_rows"].empty:
                st.info("No rows were flagged by the selected method and settings.")
            else:
                st.dataframe(outlier_result["flagged_rows"], use_container_width=True)

            labels = outlier_result["labels"]
            result_columns = [
                column
                for column in outlier_result.get("numeric_columns", [])
                if column in working_df.columns
            ]
            if result_columns:
                plot_column = st.selectbox(
                    "Variable for outlier plots",
                    result_columns,
                    key="eda_outlier_plot_column",
                    help="Choose which numeric variable to show in the outlier chart.",
                )
                lower_threshold, upper_threshold = _outlier_thresholds(outlier_result, plot_column)
                _render_eda_chart(
                    plot_outlier_boxplot(working_df, plot_column, labels),
                    title=f"Outlier boxplot for {plot_column}",
                    chart_type="outlier_chart",
                    variables_used=[plot_column],
                    method=f"{outlier_method} outlier flags with boxplot",
                    sample_size=_rows_used(working_df, [plot_column]),
                    next_step="Inspect flagged rows before deciding whether they reflect errors or valid extremes.",
                )
                _render_eda_chart(
                    plot_outlier_histogram(
                        working_df,
                        plot_column,
                        labels,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                    ),
                    title=f"Outlier histogram for {plot_column}",
                    chart_type="outlier_chart",
                    variables_used=[plot_column],
                    method=f"{outlier_method} thresholds over histogram",
                    sample_size=_rows_used(working_df, [plot_column]),
                    next_step="Use the score table and original rows to decide whether action is justified.",
                )
                if len(result_columns) >= 2:
                    scatter_x = st.selectbox(
                        "Scatter x",
                        result_columns,
                        key="eda_outlier_scatter_x",
                        help="Choose the x-axis variable for the outlier scatter plot.",
                    )
                    scatter_y_options = [column for column in result_columns if column != scatter_x]
                    scatter_y = st.selectbox(
                        "Scatter y",
                        scatter_y_options,
                        key="eda_outlier_scatter_y",
                        help="Choose the y-axis variable for the outlier scatter plot.",
                    )
                    _render_eda_chart(
                        plot_outlier_scatter(working_df, scatter_x, scatter_y, labels),
                        title=f"Outlier scatter: {scatter_x} vs {scatter_y}",
                        chart_type="outlier_chart",
                        variables_used=[scatter_x, scatter_y],
                        method=f"{outlier_method} flags on two-variable scatter",
                        sample_size=_rows_used(working_df, [scatter_x, scatter_y]),
                        next_step="Check whether flagged observations are isolated in more than one variable.",
                    )
                    _render_eda_chart(
                        plot_outlier_pca(outlier_result["pca_scores"], labels),
                        title="Outlier PCA view",
                        chart_type="outlier_chart",
                        variables_used=result_columns,
                        method=f"{outlier_method} flags projected onto PCA components",
                        sample_size=int(outlier_result["pca_scores"].shape[0]),
                        missing_handling="Uses rows available to the outlier detection result.",
                        next_step="Treat this projection as exploratory; inspect flagged rows directly.",
                    )

            st.write("Add flag column")
            new_flag_column = st.text_input(
                "New outlier flag column name",
                value="outlier_flag",
                help="Name for the new working-dataset column that stores True/False outlier flags.",
            )
            if st.button(
                "Confirm add outlier flag column",
                help="Add outlier flags to working_df only after confirmation. The original dataset is not changed.",
            ):
                try:
                    new_df, log_entry = add_outlier_flag_column(
                        working_df,
                        outlier_result,
                        new_column=new_flag_column,
                    )
                    apply_transformation_result(new_df, log_entry)
                except Exception as error:
                    st.error(f"Could not add outlier flag column: {error}")
                else:
                    st.success(f"Added `{new_flag_column}` to the working dataset.")

    st.subheader("Correlation Matrix")
    st.info(
        "Correlation is descriptive and does not imply causation. Pearson correlation may miss nonlinear relationships, "
        "and high predictor-predictor correlation may cause multicollinearity in regression models."
    )
    numeric_columns = [
        column
        for column in working_df.columns
        if _explorer_role(_detected_type_lookup(summary, column), working_df[column]) == "numeric"
    ]
    if len(numeric_columns) < 2:
        st.info("At least two numeric variables are needed for a correlation matrix.")
    else:
        selected_numeric_columns = st.multiselect(
            "Numeric variables",
            numeric_columns,
            default=numeric_columns[: min(len(numeric_columns), 8)],
            key="eda_correlation_columns",
            help="Choose numeric variables to include in the correlation matrix.",
        )
        correlation_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            help=(
                "Pearson is for approximately linear numeric relationships. "
                "Spearman is useful for monotonic or ordinal-like numeric relationships."
            ),
            key="eda_correlation_method",
        )
        correlation_threshold = st.slider(
            "Strong correlation threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="eda_correlation_threshold",
            help="Pairs with absolute correlation at or above this value are highlighted as strong.",
        )
        if len(selected_numeric_columns) < 2:
            st.info("Choose at least two numeric variables.")
        else:
            corr_matrix = build_correlation_matrix(
                working_df,
                columns=selected_numeric_columns,
                method=correlation_method,
            )
            _render_eda_chart(
                plot_correlation_heatmap(corr_matrix, threshold=correlation_threshold),
                title=f"{correlation_method.title()} correlation heatmap",
                chart_type="correlation_heatmap",
                variables_used=selected_numeric_columns,
                method=f"{correlation_method.title()} correlation",
                sample_size=_rows_used(working_df, selected_numeric_columns),
                missing_handling="Correlations use available complete pairs according to pandas correlation behavior.",
                next_step="Inspect strong pairs with the Relationship Explorer before modeling.",
            )
            pairs_table = build_correlation_pairs_table(
                working_df,
                columns=selected_numeric_columns,
                method=correlation_method,
            )
            strong_pairs = identify_high_correlation_pairs(corr_matrix, threshold=correlation_threshold)
            st.write("Pairwise correlation table")
            st.dataframe(pairs_table, use_container_width=True)
            st.write("Strong correlation pairs")
            if strong_pairs.empty:
                st.info("No variable pairs meet the selected threshold.")
            else:
                st.dataframe(strong_pairs, use_container_width=True)

    st.subheader("Relationship Explorer")
    st.info(
        "These views are exploratory only. They help inspect associations before modeling, but they do not fit models or "
        "save model runs."
    )
    relationship_columns = list(working_df.columns)
    if len(relationship_columns) < 2:
        st.info("At least two variables are needed for relationship exploration.")
    else:
        column_a = st.selectbox(
            "Variable A",
            relationship_columns,
            key="eda_relationship_a",
            help="Choose the first variable for the relationship explorer.",
        )
        column_b_options = [column for column in relationship_columns if column != column_a]
        column_b = st.selectbox(
            "Variable B",
            column_b_options,
            key="eda_relationship_b",
            help="Choose the second variable. The app selects the display based on both variable types.",
        )
        role_a = _explorer_role(_detected_type_lookup(summary, column_a), working_df[column_a])
        role_b = _explorer_role(_detected_type_lookup(summary, column_b), working_df[column_b])
        st.caption(f"Variable A role: {role_a}; Variable B role: {role_b}")

        if role_a == "numeric" and role_b == "numeric":
            show_trendline = st.checkbox(
                "Show trend line",
                value=True,
                key="eda_relationship_trendline",
                help="Adds a simple OLS trend line for visual guidance. It is not a full model result.",
            )
            st.dataframe(numeric_numeric_relationship_summary(working_df, column_a, column_b), use_container_width=True)
            _render_eda_chart(
                plot_numeric_vs_numeric_scatter(
                    working_df,
                    x_column=column_a,
                    y_column=column_b,
                    show_trendline=show_trendline,
                ),
                title=f"{column_b} vs {column_a}",
                chart_type="scatter_plot",
                variables_used=[column_a, column_b],
                method="Scatter plot with optional OLS trend line",
                sample_size=_rows_used(working_df, [column_a, column_b]),
                next_step="Review both Pearson and Spearman summaries before choosing a modeling approach.",
            )
        elif {role_a, role_b} == {"numeric", "categorical"}:
            numeric_column = column_a if role_a == "numeric" else column_b
            categorical_column = column_b if role_a == "numeric" else column_a
            st.dataframe(
                numeric_categorical_relationship_summary(
                    working_df,
                    numeric_column=numeric_column,
                    categorical_column=categorical_column,
                ),
                use_container_width=True,
            )
            _render_eda_chart(
                plot_numeric_by_category_box(
                    working_df,
                    numeric_column=numeric_column,
                    category_column=categorical_column,
                ),
                title=f"{numeric_column} by {categorical_column}",
                chart_type="grouped_boxplot",
                variables_used=[numeric_column, categorical_column],
                method="Grouped boxplot",
                sample_size=_rows_used(working_df, [numeric_column, categorical_column]),
                next_step="Compare the grouped summary table before interpreting differences.",
            )
            _render_eda_chart(
                plot_numeric_by_category_violin(
                    working_df,
                    numeric_column=numeric_column,
                    category_column=categorical_column,
                    title=f"{numeric_column} distribution by {categorical_column}",
                ),
                title=f"{numeric_column} distribution by {categorical_column}",
                chart_type="grouped_boxplot",
                variables_used=[numeric_column, categorical_column],
                method="Grouped violin plot",
                sample_size=_rows_used(working_df, [numeric_column, categorical_column]),
                next_step="Check whether group sizes are large enough for stable comparison.",
            )
        elif role_a == "categorical" and role_b == "categorical":
            tables = categorical_categorical_relationship_tables(working_df, column_a, column_b)
            st.write("Contingency table")
            st.dataframe(tables["contingency_table"], use_container_width=True)
            st.write("Row percentage table")
            st.dataframe(tables["row_percentage_table"], use_container_width=True)
            st.write("Column percentage table")
            st.dataframe(tables["column_percentage_table"], use_container_width=True)
            st.write("Chi-square test")
            st.dataframe(tables["chi_square_summary"], use_container_width=True)
            _render_eda_chart(
                plot_categorical_relationship_bar(
                    working_df,
                    x_column=column_a,
                    target_column=column_b,
                    normalize=True,
                ),
                title=f"{column_b} distribution by {column_a}",
                chart_type="categorical_bar",
                variables_used=[column_a, column_b],
                method="Normalized grouped bar chart",
                sample_size=_rows_used(working_df, [column_a, column_b]),
                next_step="Use the contingency and percentage tables to check category relationships.",
            )
        elif {role_a, role_b} == {"datetime", "numeric"}:
            datetime_column = column_a if role_a == "datetime" else column_b
            numeric_column = column_b if role_a == "datetime" else column_a
            rolling_window = st.number_input(
                "Rolling mean window",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="eda_datetime_rolling_window",
                help="Use values above 1 to smooth the time plot with a rolling mean.",
            )
            st.dataframe(
                datetime_numeric_relationship_summary(
                    working_df,
                    datetime_column=datetime_column,
                    numeric_column=numeric_column,
                ),
                use_container_width=True,
            )
            _render_eda_chart(
                plot_datetime_numeric_line(
                    working_df,
                    datetime_column=datetime_column,
                    numeric_column=numeric_column,
                    rolling_window=int(rolling_window) if rolling_window > 1 else None,
                ),
                title=f"{numeric_column} over {datetime_column}",
                chart_type="datetime_counts",
                variables_used=[datetime_column, numeric_column],
                method="Time plot with optional rolling mean",
                sample_size=_rows_used(working_df, [datetime_column, numeric_column]),
                missing_handling="Rows with invalid dates or missing numeric values are excluded.",
                next_step="Check whether trend, seasonality, or data collection timing matters.",
            )
        else:
            st.info("No relationship explorer view is available for this variable combination yet.")

    st.subheader("Target-aware EDA")
    target_column = st.selectbox(
        "Target variable",
        working_df.columns,
        key="eda_target_variable",
        help="Choose the outcome you may later model. This page only explores it; it does not fit a model.",
    )
    target_profile = build_target_profile(working_df, target_column, summary)
    target_type = target_profile["target_type"]

    st.caption(f"Target type: {target_type}")
    st.json(target_profile)

    st.write("Recommended analysis modules")
    st.dataframe(build_model_recommendations(target_type), use_container_width=True)

    feature_options = [column for column in working_df.columns if column != target_column]
    if not feature_options:
        st.info("No feature variables are available for target-aware EDA.")
    else:
        feature_column = st.selectbox(
            "Feature variable",
            feature_options,
            key="eda_feature_variable",
            help="Choose a possible predictor to compare with the selected target.",
        )
        feature_type = summary.loc[
            summary["variable"] == feature_column,
            "detected_type",
        ].iloc[0]
        feature_role = variable_role(feature_type)

        st.caption(f"Feature type: {feature_type}")

        if is_numeric_target(target_type):
            if feature_role == "numeric":
                relationship_table = numeric_target_numeric_feature_summary(
                    working_df,
                    target_column,
                    feature_column,
                )
                st.dataframe(relationship_table, use_container_width=True)
                _render_eda_chart(
                    plot_numeric_vs_numeric_scatter(working_df, feature_column, target_column),
                    title=f"{target_column} vs {feature_column}",
                    chart_type="scatter_plot",
                    variables_used=[feature_column, target_column],
                    method="Target-aware scatter plot",
                    sample_size=_rows_used(working_df, [feature_column, target_column]),
                    next_step="Use this only as exploratory guidance before model fitting.",
                )
            elif feature_role == "categorical":
                relationship_table = numeric_target_categorical_feature_summary(
                    working_df,
                    target_column,
                    feature_column,
                )
                st.dataframe(relationship_table, use_container_width=True)
                _render_eda_chart(
                    plot_numeric_by_category_box(
                        working_df,
                        numeric_column=target_column,
                        category_column=feature_column,
                        title=f"{target_column} by {feature_column}",
                    ),
                    title=f"{target_column} by {feature_column}",
                    chart_type="grouped_boxplot",
                    variables_used=[target_column, feature_column],
                    method="Target grouped by categorical feature",
                    sample_size=_rows_used(working_df, [target_column, feature_column]),
                    next_step="Review grouped mean, median, and count before modeling.",
                )
            else:
                st.info("No target-aware relationship view is available for this feature type yet.")
        elif target_type == "binary":
            target_classes = working_df[target_column].dropna().unique().tolist()
            positive_class = st.selectbox(
                "Positive class",
                target_classes,
                key="eda_target_positive_class",
                help="Choose which target class should count as the event or success case in event-rate summaries.",
            )
            if feature_role == "numeric":
                relationship_table = binary_target_numeric_feature_summary(
                    working_df,
                    target_column,
                    feature_column,
                )
                st.dataframe(relationship_table, use_container_width=True)
                _render_eda_chart(
                    plot_numeric_by_category_box(
                        working_df,
                        numeric_column=feature_column,
                        category_column=target_column,
                        title=f"{feature_column} by {target_column}",
                    ),
                    title=f"{feature_column} by {target_column}",
                    chart_type="grouped_boxplot",
                    variables_used=[feature_column, target_column],
                    method="Feature distribution by binary target",
                    sample_size=_rows_used(working_df, [feature_column, target_column]),
                    next_step="Compare groups and then consider logistic regression or ML classification.",
                )
            elif feature_role == "categorical":
                relationship_table = binary_target_categorical_feature_summary(
                    working_df,
                    target_column,
                    feature_column,
                    positive_class=positive_class,
                )
                st.dataframe(relationship_table, use_container_width=True)
                _render_eda_chart(
                    plot_categorical_relationship_bar(
                        working_df,
                        x_column=feature_column,
                        target_column=target_column,
                        normalize=True,
                    ),
                    title=f"Event rate view: {target_column} by {feature_column}",
                    chart_type="categorical_bar",
                    variables_used=[feature_column, target_column],
                    method="Normalized grouped bar chart",
                    sample_size=_rows_used(working_df, [feature_column, target_column]),
                    next_step="Use the event-rate table before choosing predictors.",
                )
            else:
                st.info("No target-aware relationship view is available for this feature type yet.")
        elif is_categorical_target(target_type):
            if feature_role == "numeric":
                relationship_table = categorical_target_numeric_feature_summary(
                    working_df,
                    target_column,
                    feature_column,
                )
                st.dataframe(relationship_table, use_container_width=True)
                _render_eda_chart(
                    plot_numeric_by_category_box(
                        working_df,
                        numeric_column=feature_column,
                        category_column=target_column,
                        title=f"{feature_column} by {target_column}",
                    ),
                    title=f"{feature_column} by {target_column}",
                    chart_type="grouped_boxplot",
                    variables_used=[feature_column, target_column],
                    method="Grouped boxplot by categorical target",
                    sample_size=_rows_used(working_df, [feature_column, target_column]),
                    next_step="Check category sizes before treating differences as meaningful.",
                )
            elif feature_role == "categorical":
                contingency_table, row_percentage_table = categorical_target_categorical_feature_tables(
                    working_df,
                    target_column,
                    feature_column,
                )
                st.write("Contingency table")
                st.dataframe(contingency_table, use_container_width=True)
                st.write("Row percentage table")
                st.dataframe(row_percentage_table, use_container_width=True)
                _render_eda_chart(
                    plot_categorical_relationship_bar(
                        working_df,
                        x_column=feature_column,
                        target_column=target_column,
                        normalize=True,
                    ),
                    title=f"{target_column} by {feature_column}",
                    chart_type="categorical_bar",
                    variables_used=[feature_column, target_column],
                    method="Normalized grouped bar chart",
                    sample_size=_rows_used(working_df, [feature_column, target_column]),
                    next_step="Use contingency tables to understand category overlap.",
                )
            else:
                st.info("No target-aware relationship view is available for this feature type yet.")
        else:
            st.info("No target-aware EDA view is available for this target type yet.")

    st.subheader("Missing Values by Variable")
    missing_table = build_missing_table(working_df)
    st.dataframe(missing_table, use_container_width=True)
    _render_eda_chart(
        plot_missing_bar(missing_table),
        title="Missing values by variable",
        chart_type="missing_bar",
        variables_used=missing_table["variable"].astype(str).tolist() if "variable" in missing_table else [],
        method="Missing count bar chart",
        sample_size=len(working_df),
        missing_handling="Counts missing values in each variable.",
        next_step="Use Data Cleaning to preview and confirm any missing-value handling.",
    )

    st.subheader("Missing Values by Row")
    st.dataframe(build_row_missing_summary(working_df), use_container_width=True)

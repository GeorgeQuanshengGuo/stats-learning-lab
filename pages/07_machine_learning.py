"""Streamlit page for machine learning baseline models."""

import pandas as pd
import streamlit as st

from src.core.model_artifacts import initialize_model_artifacts
from src.core.model_run import add_model_run_to_session, get_model_runs
from src.core.state import apply_transformation_result
from src.eda.summary import build_summary_table
from src.modeling.interpretability.calibration import (
    CALIBRATION_NOTE,
    build_binary_calibration_summary,
)
from src.modeling.machine_learning.classification import (
    SUPPORTED_CLASSIFICATION_MODELS,
    run_ml_binary_classification_models,
)
from src.modeling.machine_learning.regression import (
    SUPPORTED_REGRESSION_MODELS,
    run_ml_regression_models,
)
from src.modeling.machine_learning.multiclass_classification import (
    SUPPORTED_MULTICLASS_MODELS,
    run_ml_multiclass_classification_models,
)
from src.modeling.machine_learning.dimensionality_reduction import (
    add_pca_scores_to_dataframe,
    build_pca_component_interpretation,
    run_pca_analysis,
    save_pca_result_to_session,
)
from src.modeling.machine_learning.clustering import (
    CLUSTERING_NOTE,
    add_cluster_labels_to_dataframe,
    compute_kmeans_elbow_table,
    run_clustering_analysis,
    save_clustering_result_to_session,
)
from src.modeling.machine_learning.anomaly_detection import (
    ANOMALY_WARNING,
    add_anomaly_flag_to_dataframe,
    run_anomaly_detection,
    save_anomaly_result_to_session,
)
from src.modeling.machine_learning.tuning import (
    BINARY_TUNING_SCORING,
    REGRESSION_TUNING_SCORING,
    SUPPORTED_TUNING_CLASSIFICATION_MODELS,
    SUPPORTED_TUNING_REGRESSION_MODELS,
    run_tuned_ml_binary_classification_models,
    run_tuned_ml_regression_models,
)
from src.modeling.interpretability.coefficients import add_coefficient_interpretation_notes
from src.modeling.interpretability.tree_explainer import (
    TREE_EXPLANATION_WARNING,
    create_tree_plot_figure,
    extract_tree_feature_importance,
    extract_tree_rules,
    get_decision_path_for_observation,
    summarize_leaf_prediction,
)
from src.modeling.interpretability.pdp_ice import (
    CORRELATION_WARNING,
    ICE_NOTE,
    PDP_NOTE,
    build_ice_table,
    build_partial_dependence_table,
    build_two_feature_partial_dependence_table,
)
from src.visualization.interpretability_plots import (
    plot_calibration_curve,
    plot_ice_curves,
    plot_partial_dependence,
    plot_probability_histogram,
    plot_two_feature_pdp,
)
from src.ui.chart_cards import render_plotly_chart_card
from src.ui.page_templates import render_dataset_required_empty_state, render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles
from src.visualization.model_plots import plot_feature_importance_bar
from src.visualization.pca_plots import (
    plot_cumulative_variance,
    plot_pc1_pc2_scatter,
    plot_scree,
)
from src.visualization.clustering_plots import (
    plot_cluster_pca_scatter,
    plot_cluster_size_bar,
    plot_kmeans_elbow,
    plot_silhouette,
)
from src.visualization.anomaly_plots import (
    plot_anomaly_pca_scatter,
    plot_anomaly_score_distribution,
    plot_feature_boxplots,
)


def _model_checklist(model_names: list[str], key_prefix: str, default_models: set[str]) -> list[str]:
    """Render one checkbox per model and return selected model names."""
    selected = []
    for model_name in model_names:
        checked = st.checkbox(
            model_name,
            value=model_name in default_models,
            key=f"{key_prefix}_{model_name}",
            help="Select this model to include it in the next run.",
        )
        if checked:
            selected.append(model_name)
    return selected


def _regression_results_table(results: list[dict]) -> pd.DataFrame:
    """Build the regression results table shown on the page."""
    rows = []
    for result in results:
        model_run = result["model_run"]
        train_metrics = model_run["train_metrics"]
        test_metrics = model_run["test_metrics"]
        tuning = model_run.get("preprocessing", {}).get("tuning", {})
        rows.append(
            {
                "model_name": model_run["model_name"],
                "train_rmse": train_metrics.get("train_rmse"),
                "test_rmse": test_metrics.get("test_rmse"),
                "train_mae": train_metrics.get("train_mae"),
                "test_mae": test_metrics.get("test_mae"),
                "train_r2": train_metrics.get("train_r2"),
                "test_r2": test_metrics.get("test_r2"),
                "cv_rmse_mean": test_metrics.get("cv_rmse_mean"),
                "cv_rmse_std": test_metrics.get("cv_rmse_std"),
                "cv_mae_mean": test_metrics.get("cv_mae_mean"),
                "cv_mae_std": test_metrics.get("cv_mae_std"),
                "cv_r2_mean": test_metrics.get("cv_r2_mean"),
                "cv_r2_std": test_metrics.get("cv_r2_std"),
                "tuning_method": tuning.get("search_type"),
                "tuning_scoring": tuning.get("scoring"),
                "best_cv_score": tuning.get("best_cv_score"),
                "runtime_seconds": tuning.get("runtime_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _classification_results_table(results: list[dict]) -> pd.DataFrame:
    """Build the binary classification results table shown on the page."""
    rows = []
    for result in results:
        model_run = result["model_run"]
        train_metrics = model_run["train_metrics"]
        test_metrics = model_run["test_metrics"]
        tuning = model_run.get("preprocessing", {}).get("tuning", {})
        rows.append(
            {
                "model_name": model_run["model_name"],
                "train_accuracy": train_metrics.get("train_accuracy"),
                "test_accuracy": test_metrics.get("test_accuracy"),
                "train_precision": train_metrics.get("train_precision"),
                "test_precision": test_metrics.get("test_precision"),
                "train_recall": train_metrics.get("train_recall"),
                "test_recall": test_metrics.get("test_recall"),
                "train_f1": train_metrics.get("train_f1"),
                "test_f1": test_metrics.get("test_f1"),
                "train_roc_auc": train_metrics.get("train_roc_auc"),
                "test_roc_auc": test_metrics.get("test_roc_auc"),
                "train_pr_auc": train_metrics.get("train_pr_auc"),
                "test_pr_auc": test_metrics.get("test_pr_auc"),
                "cv_accuracy_mean": test_metrics.get("cv_accuracy_mean"),
                "cv_accuracy_std": test_metrics.get("cv_accuracy_std"),
                "cv_precision_mean": test_metrics.get("cv_precision_mean"),
                "cv_precision_std": test_metrics.get("cv_precision_std"),
                "cv_recall_mean": test_metrics.get("cv_recall_mean"),
                "cv_recall_std": test_metrics.get("cv_recall_std"),
                "cv_f1_mean": test_metrics.get("cv_f1_mean"),
                "cv_f1_std": test_metrics.get("cv_f1_std"),
                "cv_roc_auc_mean": test_metrics.get("cv_roc_auc_mean"),
                "cv_roc_auc_std": test_metrics.get("cv_roc_auc_std"),
                "tuning_method": tuning.get("search_type"),
                "tuning_scoring": tuning.get("scoring"),
                "best_cv_score": tuning.get("best_cv_score"),
                "runtime_seconds": tuning.get("runtime_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _multiclass_results_table(results: list[dict]) -> pd.DataFrame:
    """Build the multiclass classification results table shown on the page."""
    rows = []
    for result in results:
        model_run = result["model_run"]
        train_metrics = model_run["train_metrics"]
        test_metrics = model_run["test_metrics"]
        rows.append(
            {
                "model_name": model_run["model_name"],
                "train_accuracy": train_metrics.get("train_accuracy"),
                "test_accuracy": test_metrics.get("test_accuracy"),
                "train_macro_f1": train_metrics.get("train_macro_f1"),
                "test_macro_f1": test_metrics.get("test_macro_f1"),
                "train_weighted_f1": train_metrics.get("train_weighted_f1"),
                "test_weighted_f1": test_metrics.get("test_weighted_f1"),
                "train_log_loss": train_metrics.get("train_log_loss"),
                "test_log_loss": test_metrics.get("test_log_loss"),
            }
        )
    return pd.DataFrame(rows)


def _save_model_runs(results: list[dict]) -> None:
    """Save each returned ModelRun into Streamlit session state."""
    for result in results:
        add_model_run_to_session(result["model_run"])


def _feature_importance_table(result: dict) -> pd.DataFrame:
    """Return a feature-importance table from a result wrapper."""
    table = result.get("feature_importance_table")
    if isinstance(table, pd.DataFrame):
        return table
    records = result["model_run"].get("feature_importance_table") or []
    return pd.DataFrame(records)


def _importance_label(importance_type: str) -> str:
    """Return a short label for one importance type."""
    if importance_type == "tree_based":
        return "Tree-based feature importance"
    if importance_type == "permutation":
        return "Permutation importance"
    return str(importance_type)


def _render_ml_chart(
    figure,
    title: str,
    chart_type: str,
    variables_used: list[str],
    method: str,
    sample_size: int | None,
    next_step: str,
    missing_handling: str = "Uses the rows available to the displayed model or analysis result.",
) -> None:
    """Render one machine-learning chart with consistent chart-card context."""
    render_plotly_chart_card(
        figure,
        title=title,
        chart_type=chart_type,
        source_page="Machine Learning",
        variables_used=variables_used,
        method=method,
        sample_size=sample_size,
        missing_handling=missing_handling,
        next_step=next_step,
    )


def _show_feature_importance(results: list[dict]) -> None:
    """Display feature-importance tables and charts for fitted models."""
    importance_results = [
        (result["model_run"]["model_name"], _feature_importance_table(result))
        for result in results
    ]
    importance_results = [
        (model_name, table)
        for model_name, table in importance_results
        if not table.empty
    ]

    if not importance_results:
        return

    st.subheader("Feature Importance")
    for model_name, table in importance_results:
        with st.expander(model_name, expanded=len(importance_results) == 1):
            for importance_type in table["importance_type"].dropna().unique():
                subset = table.loc[table["importance_type"] == importance_type].copy()
                st.caption(_importance_label(importance_type))
                st.dataframe(subset, use_container_width=True)
                figure = plot_feature_importance_bar(
                    subset,
                    title=f"{model_name}: {_importance_label(importance_type)}",
                )
                if figure is not None:
                    _render_ml_chart(
                        figure,
                        title=f"{model_name}: {_importance_label(importance_type)}",
                        chart_type="feature_importance",
                        variables_used=subset["feature"].astype(str).tolist(),
                        method=_importance_label(importance_type),
                        sample_size=len(subset),
                        next_step="Use feature importance as a model-behavior clue, not as causal evidence.",
                    )


def _tuning_controls(task_key: str, scoring_options: list[str]) -> dict:
    """Render tuning controls and return selected options."""
    tuning_mode = st.selectbox(
        "Hyperparameter tuning",
        ["No tuning", "Tune selected models"],
        key=f"{task_key}_tuning_mode",
        help="Choose whether to run baseline models as-is or search over a small parameter grid.",
    )
    if tuning_mode == "No tuning":
        return {"enabled": False}

    search_type = st.selectbox(
        "Search type",
        ["randomized", "grid"],
        key=f"{task_key}_search_type",
        help="Randomized search samples parameter combinations; grid search tries all listed combinations.",
    )
    cv_folds = st.selectbox(
        "Tuning CV folds",
        [3, 5, 10],
        index=1,
        key=f"{task_key}_tuning_cv",
        help="Number of folds used inside the tuning search. Preprocessing stays inside the Pipeline for each fold.",
    )
    scoring = st.selectbox(
        "Tuning scoring metric",
        scoring_options,
        key=f"{task_key}_tuning_scoring",
        help="Metric used to choose the best parameter settings.",
    )
    n_iter = 10
    if search_type == "randomized":
        n_iter = st.slider(
            "Random search iterations",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
            key=f"{task_key}_n_iter",
            help="Number of random parameter combinations to try.",
        )
    st.caption("Tuning searches the full sklearn Pipeline, so preprocessing is fit inside each CV fold.")
    return {
        "enabled": True,
        "search_type": search_type,
        "cv_folds": int(cv_folds),
        "scoring": scoring,
        "n_iter": int(n_iter),
    }


def _show_tuning_results(results: list[dict]) -> None:
    """Display tuning metadata and cv_results tables when available."""
    tuned_results = [
        result
        for result in results
        if result["model_run"].get("preprocessing", {}).get("tuning", {}).get("enabled")
    ]
    if not tuned_results:
        return

    st.subheader("Hyperparameter Tuning Details")
    for result in tuned_results:
        model_run = result["model_run"]
        tuning = model_run.get("preprocessing", {}).get("tuning", {})
        with st.expander(model_run["model_name"], expanded=len(tuned_results) == 1):
            st.json(
                {
                    "search_type": tuning.get("search_type"),
                    "cv_folds": tuning.get("cv_folds"),
                    "scoring": tuning.get("scoring"),
                    "best_cv_score": tuning.get("best_cv_score"),
                    "best_params": tuning.get("best_params"),
                    "runtime_seconds": tuning.get("runtime_seconds"),
                }
            )
            if tuning.get("overfitting_warning"):
                st.warning(tuning["overfitting_warning"])
            table = result.get("tuning_results_table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.dataframe(table, use_container_width=True)


def _show_multiclass_details(results: list[dict]) -> None:
    """Display multiclass confusion matrices and class-wise metric tables."""
    if not results:
        return
    st.subheader("Multiclass Diagnostics")
    for result in results:
        model_name = result["model_run"]["model_name"]
        with st.expander(model_name, expanded=len(results) == 1):
            st.caption("Confusion matrix")
            st.dataframe(result["confusion_matrix"], use_container_width=True)
            st.caption("Class-wise metrics")
            st.dataframe(result["class_wise_metrics"], use_container_width=True)
            probability_table = result.get("probability_table")
            if isinstance(probability_table, pd.DataFrame) and not probability_table.empty:
                st.caption("Predicted probability table")
                st.dataframe(probability_table.head(50), use_container_width=True)


def _show_pca_result(result: dict, working_df: pd.DataFrame, color_column: str | None) -> None:
    """Display PCA tables and plots."""
    st.subheader("PCA Results")
    st.info(result["notes"])

    st.caption("Explained variance")
    st.dataframe(result["explained_variance_table"], use_container_width=True)

    st.caption("Component loadings")
    st.dataframe(result["loadings_table"], use_container_width=True)
    interpretation_table = build_pca_component_interpretation(result["loadings_table"])
    if not interpretation_table.empty:
        st.caption("What PC1 / PC2 may roughly represent")
        st.info(
            "Large absolute loadings show which original variables mostly shape each component. "
            "The sign direction in PCA can flip, so use this as a naming clue rather than a fixed meaning."
        )
        st.dataframe(interpretation_table, use_container_width=True)

    st.caption("PCA scores")
    st.dataframe(result["scores"], use_container_width=True)

    scree_figure = plot_scree(result["explained_variance_table"])
    if scree_figure is not None:
        _render_ml_chart(
            scree_figure,
            title="PCA scree plot",
            chart_type="pca_scree",
            variables_used=result.get("feature_columns", []),
            method="PCA explained variance by component",
            sample_size=int(result["scores"].shape[0]),
            next_step="Use the cumulative variance chart to decide how many components are useful.",
        )

    cumulative_figure = plot_cumulative_variance(result["explained_variance_table"])
    if cumulative_figure is not None:
        _render_ml_chart(
            cumulative_figure,
            title="PCA cumulative explained variance",
            chart_type="pca_scree",
            variables_used=result.get("feature_columns", []),
            method="Cumulative PCA explained variance",
            sample_size=int(result["scores"].shape[0]),
            next_step="Choose a component count based on explained variance and analysis needs.",
        )

    color_values = None
    if color_column and color_column in working_df.columns:
        color_values = working_df.loc[result["scores"].index, color_column]
    scatter_figure = plot_pc1_pc2_scatter(result["scores"], color_values=color_values)
    if scatter_figure is not None:
        _render_ml_chart(
            scatter_figure,
            title="PCA PC1 vs PC2 scatter",
            chart_type="pca_scatter",
            variables_used=result.get("feature_columns", []),
            method="PCA projection onto first two components",
            sample_size=int(result["scores"].shape[0]),
            next_step="Use the scatter as exploratory structure, not as proof of true groups.",
        )
    else:
        st.info("PC1 vs PC2 scatter requires at least two PCA components.")


def _show_clustering_result(result: dict) -> None:
    """Display clustering tables and plots."""
    st.subheader("Clustering Results")
    st.info(result.get("notes") or CLUSTERING_NOTE)

    metrics = result.get("metrics") or {}
    metric_columns = st.columns(3)
    metric_columns[0].metric("Silhouette", _metric_display(metrics.get("silhouette_score")))
    metric_columns[1].metric("Davies-Bouldin", _metric_display(metrics.get("davies_bouldin_score")))
    metric_columns[2].metric("Calinski-Harabasz", _metric_display(metrics.get("calinski_harabasz_score")))

    st.caption("Cluster labels")
    st.info(
        "Cluster labels are algorithm-assigned group names. Label 0, 1, or 2 is not a rank, score, or quality level; "
        "use the cluster summaries below to decide whether a label has a useful real-world description."
    )
    st.dataframe(result["cluster_labels"].rename("cluster_label").to_frame(), use_container_width=True)

    st.caption("Cluster sizes")
    st.dataframe(result["cluster_size_table"], use_container_width=True)
    size_figure = plot_cluster_size_bar(result["cluster_size_table"])
    if size_figure is not None:
        _render_ml_chart(
            size_figure,
            title="Cluster sizes",
            chart_type="cluster_chart",
            variables_used=result.get("feature_columns", []),
            method="Cluster label count",
            sample_size=int(result["cluster_labels"].shape[0]),
            next_step="Check whether clusters are too small or too imbalanced to be useful.",
        )

    st.caption("Cluster summary by feature")
    st.dataframe(result["cluster_summary_table"], use_container_width=True)

    scatter_figure = plot_cluster_pca_scatter(result["pca_scores"], result["cluster_labels"])
    if scatter_figure is not None:
        _render_ml_chart(
            scatter_figure,
            title="Cluster PCA scatter",
            chart_type="cluster_chart",
            variables_used=result.get("feature_columns", []),
            method="PCA projection colored by cluster label",
            sample_size=int(result["cluster_labels"].shape[0]),
            next_step="Compare this view with cluster summaries before naming clusters.",
        )

    elbow_table = result.get("elbow_table")
    if isinstance(elbow_table, pd.DataFrame) and not elbow_table.empty:
        st.caption("KMeans elbow table")
        st.dataframe(elbow_table, use_container_width=True)
        elbow_figure = plot_kmeans_elbow(elbow_table)
        if elbow_figure is not None:
            _render_ml_chart(
                elbow_figure,
                title="KMeans elbow plot",
                chart_type="cluster_chart",
                variables_used=result.get("feature_columns", []),
                method="KMeans inertia across k values",
                sample_size=int(result["cluster_labels"].shape[0]),
                next_step="Look for diminishing returns rather than an automatic best k.",
            )

    silhouette_table = result.get("silhouette_table")
    silhouette_figure = plot_silhouette(silhouette_table)
    if silhouette_figure is not None:
        st.caption("Silhouette plot")
        _render_ml_chart(
            silhouette_figure,
            title="Silhouette plot",
            chart_type="cluster_chart",
            variables_used=result.get("feature_columns", []),
            method="Per-row silhouette values",
            sample_size=int(silhouette_table.shape[0]) if isinstance(silhouette_table, pd.DataFrame) else None,
            next_step="Inspect low or negative silhouette values for weak cluster separation.",
        )


def _show_anomaly_result(result: dict) -> None:
    """Display anomaly detection tables and plots."""
    st.subheader("Anomaly Detection Results")
    st.warning(result.get("notes") or ANOMALY_WARNING)

    summary = result.get("summary") or {}
    metric_columns = st.columns(3)
    metric_columns[0].metric("Anomalies", summary.get("number_of_anomalies", 0))
    metric_columns[1].metric("Anomaly %", f"{summary.get('anomaly_percentage', 0.0):.2%}")
    metric_columns[2].metric("Rows evaluated", summary.get("rows_evaluated", 0))

    st.caption("Anomaly labels and scores")
    st.dataframe(result["result_table"], use_container_width=True)

    st.caption("Top anomalous rows")
    top_anomalies = result.get("top_anomalies")
    if isinstance(top_anomalies, pd.DataFrame) and not top_anomalies.empty:
        st.dataframe(top_anomalies, use_container_width=True)
    else:
        st.info("No anomalous rows were flagged with the current settings.")

    scatter_figure = plot_anomaly_pca_scatter(result["pca_scores"], result["anomaly_labels"])
    if scatter_figure is not None:
        _render_ml_chart(
            scatter_figure,
            title="Anomaly PCA scatter",
            chart_type="anomaly_chart",
            variables_used=result.get("feature_columns", []),
            method="PCA projection with anomaly labels",
            sample_size=int(result["anomaly_labels"].shape[0]),
            next_step="Inspect top anomalous rows before deciding whether they are errors.",
        )

    score_figure = plot_anomaly_score_distribution(result["result_table"])
    if score_figure is not None:
        _render_ml_chart(
            score_figure,
            title="Anomaly score distribution",
            chart_type="anomaly_chart",
            variables_used=result.get("feature_columns", []),
            method="Anomaly score histogram",
            sample_size=int(result["result_table"].shape[0]),
            next_step="Use scores to prioritize row review, not automatic deletion.",
        )

    boxplot_figure = plot_feature_boxplots(result["feature_boxplot_data"])
    if boxplot_figure is not None:
        _render_ml_chart(
            boxplot_figure,
            title="Feature boxplots by anomaly status",
            chart_type="anomaly_chart",
            variables_used=result.get("feature_columns", []),
            method="Boxplots split by anomaly flag",
            sample_size=int(result["feature_boxplot_data"].shape[0]),
            next_step="Compare feature ranges for flagged and typical rows.",
        )


def _metric_display(value) -> str:
    """Return a compact metric display string."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def _show_model_formulas(results: list[dict]) -> None:
    """Display fitted formulas for interpretable ML models when available."""
    formula_results = [
        (result["model_run"]["model_name"], result["model_run"].get("formula_latex"))
        for result in results
        if result["model_run"].get("formula_latex")
    ]
    if not formula_results:
        return

    st.subheader("Model Formula")
    for model_name, formula_latex in formula_results:
        with st.expander(model_name, expanded=len(formula_results) == 1):
            for key in ["estimated", "probability"]:
                formula = formula_latex.get(key)
                if formula:
                    st.latex(formula)
            if formula_latex.get("note"):
                st.caption(formula_latex["note"])


def _show_coefficient_tables(results: list[dict]) -> None:
    """Display coefficient tables and notes for interpretable ML models."""
    coefficient_results = []
    for result in results:
        model_run = result["model_run"]
        records = model_run.get("coefficient_table") or []
        table = pd.DataFrame(records)
        if not table.empty:
            coefficient_results.append((model_run["model_name"], model_run, table))

    if not coefficient_results:
        return

    st.subheader("Coefficient Summary")
    for model_name, model_run, table in coefficient_results:
        with st.expander(model_name, expanded=len(coefficient_results) == 1):
            st.dataframe(table, use_container_width=True)
            notes = add_coefficient_interpretation_notes(
                table,
                scale_numeric=bool(model_run.get("preprocessing", {}).get("scale_numeric")),
            )
            for note in notes:
                st.caption(note)


def _show_tree_explanations(results: list[dict]) -> None:
    """Display rules, plots, and example paths for single decision tree models."""
    tree_results = [
        result
        for result in results
        if result["model_run"]["model_name"] in {"Decision Tree Regressor", "Decision Tree Classifier"}
    ]
    if not tree_results:
        return

    st.subheader("Decision Tree Explanation")
    st.warning(TREE_EXPLANATION_WARNING)
    for result in tree_results:
        model_name = result["model_run"]["model_name"]
        pipeline = result["pipeline"]
        settings = result["model_run"].get("preprocessing", {}).get("decision_tree_settings") or {}
        max_depth = settings.get("max_depth") or 4

        with st.expander(model_name, expanded=len(tree_results) == 1):
            st.caption("Text rules")
            st.code(result.get("tree_rules") or extract_tree_rules(pipeline, max_depth=max_depth), language="text")

            importance_table = result.get("tree_feature_importance_table")
            if importance_table is None or importance_table.empty:
                importance_table = extract_tree_feature_importance(pipeline)
            if not importance_table.empty:
                st.caption("Tree feature importance")
                st.dataframe(importance_table, use_container_width=True)

            figure = create_tree_plot_figure(pipeline, max_depth=max_depth)
            if figure is not None:
                st.caption("Tree plot")
                _render_ml_chart(
                    figure,
                    title=f"{model_name} tree plot",
                    chart_type="decision_tree",
                    variables_used=list(result["model_run"].get("features") or []),
                    method=f"Decision tree plot limited to depth {max_depth}",
                    sample_size=None,
                    next_step="Use shallow tree views for explanation and test metrics for reliability.",
                )

            x_test = result.get("x_test")
            if isinstance(x_test, pd.DataFrame) and not x_test.empty:
                example_row = x_test.iloc[[0]]
                st.caption("Example prediction path for the first test row")
                path_table = get_decision_path_for_observation(pipeline, example_row)
                if path_table.empty:
                    st.info("This observation reached a leaf without additional split rules.")
                else:
                    st.dataframe(path_table, use_container_width=True)
                st.caption("Leaf prediction summary")
                st.json(summarize_leaf_prediction(pipeline, example_row))


def _run_label(model_run: dict) -> str:
    """Return a compact label for selecting saved model runs."""
    return (
        f"{model_run.get('model_name')} | {model_run.get('target')} | "
        f"{model_run.get('task_type')} | {model_run.get('run_id', '')[:8]}"
    )


def _artifact_table(artifact: dict, key: str) -> pd.DataFrame:
    """Return a DataFrame table saved inside a model artifact."""
    table = artifact.get(key)
    if isinstance(table, pd.DataFrame):
        return table
    return pd.DataFrame(table or [])


def _render_saved_model_interpretation(working_df: pd.DataFrame) -> None:
    """Render advanced inspection plots for saved ML model artifacts."""
    st.header("Saved Model Interpretation")
    st.caption(
        "These plots inspect fitted ML models. They describe model behavior, not causal effects."
    )

    model_runs = [
        run
        for run in get_model_runs()
        if run.get("model_family") == "machine_learning"
    ]
    model_artifacts = initialize_model_artifacts()
    usable_runs = [
        run
        for run in model_runs
        if model_artifacts.get(run.get("run_id"), {}).get("fitted_pipeline") is not None
    ]

    if not usable_runs:
        st.info("Run and save at least one machine learning model to use interpretation plots.")
        return

    labels = [_run_label(run) for run in usable_runs]
    selected_label = st.selectbox(
        "Saved ML model run",
        labels,
        key="ml_interpretation_run",
        help="Choose a saved machine learning run with a fitted Pipeline artifact.",
    )
    selected_run = usable_runs[labels.index(selected_label)]
    artifact = model_artifacts[selected_run["run_id"]]
    pipeline = artifact["fitted_pipeline"]
    features = list(artifact.get("features") or selected_run.get("features") or [])
    task_type = selected_run.get("task_type")

    st.json(
        {
            "model_name": selected_run.get("model_name"),
            "task_type": task_type,
            "target": selected_run.get("target"),
            "features": features,
        }
    )

    method_options = ["Permutation importance"]
    if task_type in {"regression", "binary_classification"}:
        method_options.extend(["PDP", "ICE"])
    if task_type == "binary_classification":
        method_options.append("Calibration")
    method = st.selectbox(
        "Interpretation method",
        method_options,
        key="ml_interpretation_method",
        help="Choose a model inspection view. These describe model behavior, not causal effects.",
    )

    x_data = working_df[features].copy(deep=True)
    if method == "Permutation importance":
        table = _artifact_table(artifact, "feature_importance_table")
        if "importance_type" in table.columns:
            table = table.loc[table["importance_type"] == "permutation"]
        else:
            table = pd.DataFrame()
        if table.empty:
            st.info("No permutation importance was saved for this run. Re-run the model with permutation importance enabled.")
            return
        st.dataframe(table, use_container_width=True)
        figure = plot_feature_importance_bar(table, title="Permutation Importance")
        if figure is not None:
            _render_ml_chart(
                figure,
                title="Permutation importance",
                chart_type="feature_importance",
                variables_used=features,
                method="Permutation importance saved with the model artifact",
                sample_size=len(table),
                next_step="Review importance together with correlated features and PDP or ICE.",
            )
        return

    if method in {"PDP", "ICE"}:
        feature = st.selectbox(
            "Feature",
            features,
            key=f"ml_{method.lower()}_feature",
            help="Choose the feature whose effect on model predictions you want to inspect.",
        )
        grid_resolution = st.slider(
            "Grid resolution",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            key=f"ml_{method.lower()}_grid",
            help="Number of feature values to evaluate. Higher values are smoother but slower.",
        )
        st.info(CORRELATION_WARNING)

        if method == "PDP":
            second_feature_options = ["None", *[column for column in features if column != feature]]
            second_feature = st.selectbox(
                "Second feature for 2D PDP",
                second_feature_options,
                key="ml_pdp_second_feature",
                help="Optional second feature for a two-feature partial dependence view.",
            )
            if second_feature == "None":
                pdp_table = build_partial_dependence_table(
                    pipeline,
                    x_data,
                    feature=feature,
                    task_type=task_type,
                    grid_resolution=grid_resolution,
                )
                st.caption(PDP_NOTE)
                st.dataframe(pdp_table, use_container_width=True)
                figure = plot_partial_dependence(pdp_table, title=f"PDP: {feature}")
            else:
                pdp_table = build_two_feature_partial_dependence_table(
                    pipeline,
                    x_data,
                    feature_a=feature,
                    feature_b=second_feature,
                    task_type=task_type,
                    grid_resolution=min(grid_resolution, 12),
                )
                st.caption(PDP_NOTE)
                st.dataframe(pdp_table, use_container_width=True)
                figure = plot_two_feature_pdp(pdp_table, title=f"2D PDP: {feature} and {second_feature}")
            if figure is not None:
                if second_feature == "None":
                    chart_title = f"PDP: {feature}"
                    variables = [feature]
                    chart_method = "Partial dependence"
                else:
                    chart_title = f"2D PDP: {feature} and {second_feature}"
                    variables = [feature, second_feature]
                    chart_method = "Two-feature partial dependence"
                _render_ml_chart(
                    figure,
                    title=chart_title,
                    chart_type="pdp",
                    variables_used=variables,
                    method=chart_method,
                    sample_size=len(x_data),
                    next_step="Use caution if selected features are strongly correlated.",
                )
            return

        ice_table = build_ice_table(
            pipeline,
            x_data,
            feature=feature,
            task_type=task_type,
            grid_resolution=grid_resolution,
        )
        st.caption(ICE_NOTE)
        st.dataframe(ice_table, use_container_width=True)
        figure = plot_ice_curves(ice_table, title=f"ICE: {feature}")
        if figure is not None:
            _render_ml_chart(
                figure,
                title=f"ICE: {feature}",
                chart_type="ice",
                variables_used=[feature],
                method="Individual conditional expectation curves",
                sample_size=len(x_data),
                next_step="Compare individual curves with the average PDP pattern.",
            )
        return

    if method == "Calibration":
        target = selected_run.get("target")
        calibration_data = working_df[[target, *features]].dropna(subset=[target]).copy(deep=True)
        positive_class = artifact.get("positive_class") or selected_run.get("preprocessing", {}).get("positive_class", 1)
        n_bins = st.slider(
            "Calibration bins",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            help="Number of probability groups used to compare predicted probabilities with observed rates.",
        )
        summary = build_binary_calibration_summary(
            pipeline,
            calibration_data[features],
            calibration_data[target],
            positive_label=positive_class,
            n_bins=n_bins,
        )
        st.caption(CALIBRATION_NOTE)
        st.metric("Brier score", f"{summary['brier_score']:.4f}")
        st.dataframe(summary["calibration_table"], use_container_width=True)
        calibration_figure = plot_calibration_curve(summary["calibration_table"])
        if calibration_figure is not None:
            _render_ml_chart(
                calibration_figure,
                title="Calibration curve",
                chart_type="calibration_plot",
                variables_used=[target],
                method="Observed frequency by predicted probability bin",
                sample_size=len(calibration_data),
                next_step="Use calibration before relying on predicted probabilities for decisions.",
            )
        histogram_figure = plot_probability_histogram(summary["probability_histogram"])
        if histogram_figure is not None:
            _render_ml_chart(
                histogram_figure,
                title="Predicted probability histogram",
                chart_type="probability_distribution",
                variables_used=[target],
                method="Predicted probability bin counts",
                sample_size=len(calibration_data),
                next_step="Review whether probabilities are concentrated near 0, 1, or the threshold.",
            )


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Machine Learning",
    purpose="Run leakage-safe sklearn baseline models, cross-validation, tuning, and model inspection tools.",
    when_to_use="Use this page when your goal is predictive performance with explicit user-controlled model choices.",
    common_mistake="Do not fit preprocessing outside the pipeline or compare train metrics without test or validation metrics.",
    key_terms=["train_test_split", "overfitting", "cross_validation", "feature_importance"],
    next_step="Save useful runs, then compare them or inspect diagnostics.",
    next_page="pages/06_model_comparison.py",
    tags=["Prediction", "Scikit-learn"],
)

working_df = st.session_state.get("working_df")

if working_df is None:
    render_dataset_required_empty_state()
else:
    summary = build_summary_table(working_df)

    task_type = st.radio(
        "Task type",
        [
            "Regression",
            "Binary Classification",
            "Multiclass Classification",
            "PCA / Dimension Reduction",
            "Clustering",
            "Anomaly Detection",
        ],
        horizontal=True,
        key="ml_task_type",
        help="Choose the kind of analysis to run. Modeling tasks save ModelRuns; PCA, clustering, and anomaly detection are exploratory.",
    )

    if task_type in {"PCA / Dimension Reduction", "Clustering", "Anomaly Detection"}:
        scale_numeric = st.checkbox(
            "Scale numeric variables",
            value=True,
            key="ml_unsupervised_scale_numeric",
            help="Put numeric features on a comparable scale. This is usually helpful for PCA, clustering, and distance-based methods.",
        )
    else:
        test_size = st.slider(
            "Test split ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="ml_test_size",
            help="Fraction of rows held out for final test metrics. Preprocessing is fit only on training data.",
        )
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="ml_random_state",
            help="Keeps train/test split and randomized models reproducible.",
        )
        scale_numeric = st.checkbox(
            "Scale numeric variables",
            value=task_type in {"Binary Classification", "Multiclass Classification"},
            key="ml_supervised_scale_numeric",
            help="Standardize numeric features inside the sklearn Pipeline. Useful for linear and distance-based models.",
        )
        cv_option = st.selectbox(
            "Cross-validation",
            ["No cross-validation", "5-fold cross-validation", "10-fold cross-validation"],
            key="ml_cv_option",
            help="Optionally evaluate models across multiple training/validation folds. The full Pipeline is used inside each fold.",
        )
        cv_folds = {
            "No cross-validation": None,
            "5-fold cross-validation": 5,
            "10-fold cross-validation": 10,
        }[cv_option]
        compute_permutation_importance = st.checkbox(
            "Compute permutation importance",
            value=False,
            key="ml_compute_permutation_importance",
            help="Optional and slower. Measures how test performance changes when a feature is shuffled.",
        )
        with st.expander("Decision tree interpretability settings"):
            st.warning(TREE_EXPLANATION_WARNING)
            tree_max_depth = st.slider(
                "Tree max depth",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                key="ml_tree_max_depth",
                help="Limits tree depth. Shallower trees are usually easier to interpret.",
            )
            tree_min_samples_leaf = st.number_input(
                "Minimum samples per leaf",
                min_value=1,
                value=1,
                step=1,
                key="ml_tree_min_samples_leaf",
                help="Minimum rows required in a final tree leaf. Larger values can reduce overfitting.",
            )
            tree_min_samples_split = st.number_input(
                "Minimum samples required to split",
                min_value=2,
                value=2,
                step=1,
                key="ml_tree_min_samples_split",
                help="Minimum rows required before a tree node can split. Larger values make trees simpler.",
            )

    if task_type == "Regression":
        numeric_target_options = summary.loc[
            summary["detected_type"].isin(["continuous_numeric", "discrete_numeric"]),
            "variable",
        ].tolist()

        if not numeric_target_options:
            st.warning("No numeric target variables were detected for regression.")
        else:
            target_column = st.selectbox(
                "Target variable",
                numeric_target_options,
                key="ml_regression_target",
                help="Choose the numeric outcome the regression models should predict.",
            )
            feature_options = [column for column in working_df.columns if column != target_column]
            feature_columns = st.multiselect(
                "Feature variables",
                feature_options,
                key="ml_regression_features",
                help="Choose predictor columns. Preprocessing for these features is fit inside each model Pipeline.",
            )

            st.subheader("Models")
            tuning_options = _tuning_controls(
                "ml_regression",
                scoring_options=list(REGRESSION_TUNING_SCORING.keys()),
            )
            available_models = (
                SUPPORTED_TUNING_REGRESSION_MODELS
                if tuning_options["enabled"]
                else SUPPORTED_REGRESSION_MODELS
            )
            selected_models = _model_checklist(
                available_models,
                "ml_regression_model",
                default_models={"Ridge"} if tuning_options["enabled"] else {"Linear Regression"},
            )

            if st.button(
                "Run regression baselines",
                help="Train the selected sklearn regression models and save each result for comparison.",
            ):
                if not feature_columns:
                    st.error("Choose at least one feature variable.")
                elif not selected_models:
                    st.error("Choose at least one model.")
                else:
                    try:
                        if tuning_options["enabled"]:
                            results = run_tuned_ml_regression_models(
                                working_df,
                                target_column=target_column,
                                feature_columns=feature_columns,
                                selected_models=selected_models,
                                test_size=float(test_size),
                                random_state=int(random_state),
                                scale_numeric=scale_numeric,
                                search_type=tuning_options["search_type"],
                                cv_folds=tuning_options["cv_folds"],
                                scoring=tuning_options["scoring"],
                                n_iter=tuning_options["n_iter"],
                            )
                        else:
                            results = run_ml_regression_models(
                                working_df,
                                target_column=target_column,
                                feature_columns=feature_columns,
                                selected_models=selected_models,
                                test_size=float(test_size),
                                random_state=int(random_state),
                                scale_numeric=scale_numeric,
                                cv_folds=cv_folds,
                                compute_permutation_importance=compute_permutation_importance,
                                tree_max_depth=int(tree_max_depth),
                                tree_min_samples_leaf=int(tree_min_samples_leaf),
                                tree_min_samples_split=int(tree_min_samples_split),
                            )
                    except Exception as error:
                        st.error(f"Could not run regression baselines: {error}")
                    else:
                        _save_model_runs(results)
                        st.session_state["latest_ml_regression_results"] = results
                        st.success("Regression results were saved for Model Comparison.")

            results = st.session_state.get("latest_ml_regression_results", [])
            if results:
                st.subheader("Regression Results")
                st.dataframe(_regression_results_table(results), use_container_width=True)
                _show_tuning_results(results)
                _show_coefficient_tables(results)
                _show_model_formulas(results)
                _show_tree_explanations(results)
                _show_feature_importance(results)

    elif task_type == "Binary Classification":
        binary_target_options = summary.loc[
            summary["detected_type"] == "binary",
            "variable",
        ].tolist()

        if not binary_target_options:
            st.warning("No binary target variables were detected for binary classification.")
        else:
            target_column = st.selectbox(
                "Target variable",
                binary_target_options,
                key="ml_classification_target",
                help="Choose a binary outcome with exactly two classes.",
            )
            target_classes = working_df[target_column].dropna().unique().tolist()
            positive_class = st.selectbox(
                "Positive class",
                target_classes,
                key="ml_positive_class",
                help="Choose the event class used for precision, recall, F1, ROC AUC, and PR AUC.",
            )
            feature_options = [column for column in working_df.columns if column != target_column]
            feature_columns = st.multiselect(
                "Feature variables",
                feature_options,
                key="ml_classification_features",
                help="Choose predictor columns. Preprocessing for these features is fit inside each model Pipeline.",
            )

            st.subheader("Models")
            tuning_options = _tuning_controls(
                "ml_classification",
                scoring_options=list(BINARY_TUNING_SCORING.keys()),
            )
            available_models = (
                SUPPORTED_TUNING_CLASSIFICATION_MODELS
                if tuning_options["enabled"]
                else SUPPORTED_CLASSIFICATION_MODELS
            )
            selected_models = _model_checklist(
                available_models,
                "ml_classification_model",
                default_models={"Logistic Regression"},
            )

            if st.button(
                "Run binary classification baselines",
                help="Train the selected sklearn binary classifiers and save each result for comparison.",
            ):
                if not feature_columns:
                    st.error("Choose at least one feature variable.")
                elif not selected_models:
                    st.error("Choose at least one model.")
                else:
                    try:
                        if tuning_options["enabled"]:
                            results = run_tuned_ml_binary_classification_models(
                                working_df,
                                target_column=target_column,
                                feature_columns=feature_columns,
                                selected_models=selected_models,
                                positive_class=positive_class,
                                test_size=float(test_size),
                                random_state=int(random_state),
                                scale_numeric=scale_numeric,
                                search_type=tuning_options["search_type"],
                                cv_folds=tuning_options["cv_folds"],
                                scoring=tuning_options["scoring"],
                                n_iter=tuning_options["n_iter"],
                            )
                        else:
                            results = run_ml_binary_classification_models(
                                working_df,
                                target_column=target_column,
                                feature_columns=feature_columns,
                                selected_models=selected_models,
                                positive_class=positive_class,
                                test_size=float(test_size),
                                random_state=int(random_state),
                                scale_numeric=scale_numeric,
                                cv_folds=cv_folds,
                                compute_permutation_importance=compute_permutation_importance,
                                tree_max_depth=int(tree_max_depth),
                                tree_min_samples_leaf=int(tree_min_samples_leaf),
                                tree_min_samples_split=int(tree_min_samples_split),
                            )
                    except Exception as error:
                        st.error(f"Could not run binary classification baselines: {error}")
                    else:
                        _save_model_runs(results)
                        st.session_state["latest_ml_classification_results"] = results
                        st.success("Binary classification results were saved for Model Comparison.")

            results = st.session_state.get("latest_ml_classification_results", [])
            if results:
                st.subheader("Binary Classification Results")
                st.dataframe(_classification_results_table(results), use_container_width=True)
                _show_tuning_results(results)
                _show_coefficient_tables(results)
                _show_model_formulas(results)
                _show_tree_explanations(results)
                _show_feature_importance(results)

    elif task_type == "Multiclass Classification":
        multiclass_target_options = summary.loc[
            summary["detected_type"].isin(["nominal_categorical", "ordinal_categorical_candidate"]),
            "variable",
        ].tolist()
        multiclass_target_options = [
            column
            for column in multiclass_target_options
            if working_df[column].dropna().nunique() > 2
        ]

        if not multiclass_target_options:
            st.warning("No target variables with three or more classes were detected.")
        else:
            target_column = st.selectbox(
                "Target variable",
                multiclass_target_options,
                key="ml_multiclass_target",
                help="Choose an unordered target with three or more classes.",
            )
            st.info(
                "Use this for unordered classes. Do not use ordered categories as multiclass unless you confirm "
                "they should be treated as unordered."
            )
            confirm_unordered = st.checkbox(
                "I confirm this target should be treated as unordered multiclass",
                value=False,
                help="Confirm only if the classes do not have a meaningful order.",
            )
            feature_options = [column for column in working_df.columns if column != target_column]
            feature_columns = st.multiselect(
                "Feature variables",
                feature_options,
                key="ml_multiclass_features",
                help="Choose predictors for multiclass classification.",
            )

            st.subheader("Models")
            selected_models = _model_checklist(
                SUPPORTED_MULTICLASS_MODELS,
                "ml_multiclass_model",
                default_models={"Multinomial Logistic Regression"},
            )

            if st.button(
                "Run multiclass classification baselines",
                help="Train selected multiclass classifiers and save each result for comparison.",
            ):
                if not confirm_unordered:
                    st.error("Confirm that the target should be treated as unordered multiclass.")
                elif not feature_columns:
                    st.error("Choose at least one feature variable.")
                elif not selected_models:
                    st.error("Choose at least one model.")
                else:
                    try:
                        results = run_ml_multiclass_classification_models(
                            working_df,
                            target_column=target_column,
                            feature_columns=feature_columns,
                            selected_models=selected_models,
                            test_size=float(test_size),
                            random_state=int(random_state),
                            scale_numeric=scale_numeric,
                            compute_permutation_importance=compute_permutation_importance,
                            tree_max_depth=int(tree_max_depth),
                            tree_min_samples_leaf=int(tree_min_samples_leaf),
                            tree_min_samples_split=int(tree_min_samples_split),
                        )
                    except Exception as error:
                        st.error(f"Could not run multiclass classification baselines: {error}")
                    else:
                        _save_model_runs(results)
                        st.session_state["latest_ml_multiclass_results"] = results
                        st.success("Multiclass classification results were saved for Model Comparison.")

            results = st.session_state.get("latest_ml_multiclass_results", [])
            if results:
                st.subheader("Multiclass Classification Results")
                st.dataframe(_multiclass_results_table(results), use_container_width=True)
                _show_multiclass_details(results)
                _show_feature_importance(results)

    elif task_type == "PCA / Dimension Reduction":
        numeric_feature_options = [
            column
            for column in working_df.columns
            if pd.api.types.is_numeric_dtype(working_df[column])
        ]

        st.info(
            "PCA is an unsupervised exploratory method for numeric features. Components summarize variation; "
            "do not interpret them as causal explanations."
        )

        if len(numeric_feature_options) < 2:
            st.warning("PCA needs at least two numeric features for a useful PC1 vs PC2 view.")
        else:
            feature_columns = st.multiselect(
                "Numeric features",
                numeric_feature_options,
                default=numeric_feature_options[: min(4, len(numeric_feature_options))],
                key="pca_feature_columns",
                help="Choose numeric variables used to compute principal components.",
            )

            if len(feature_columns) < 2:
                st.warning("Choose at least two numeric features.")
            else:
                max_components = min(len(feature_columns), len(working_df))
                if max_components < 2:
                    st.warning("PCA needs at least two rows and two numeric features for a PC1 vs PC2 view.")
                else:
                    n_components = st.slider(
                        "Number of components",
                        min_value=2,
                        max_value=max_components,
                        value=min(2, max_components),
                        step=1,
                        key="pca_n_components",
                        help="Number of PCA components to compute and display.",
                    )
                    categorical_color_options = summary.loc[
                        summary["detected_type"].isin(
                            ["binary", "nominal_categorical", "ordinal_categorical_candidate"]
                        ),
                        "variable",
                    ].tolist()
                    color_options = [
                        "None",
                        *[column for column in categorical_color_options if column not in feature_columns],
                    ]
                    color_choice = st.selectbox(
                        "Color PC1 vs PC2 by",
                        color_options,
                        key="pca_color_column",
                        help="Optional categorical variable used only to color the PCA scatter plot.",
                    )
                    color_column = None if color_choice == "None" else color_choice

                    if st.button(
                        "Run PCA",
                        help="Run PCA on selected numeric features. The working dataset is not changed.",
                    ):
                        try:
                            pca_result = run_pca_analysis(
                                working_df,
                                feature_columns=feature_columns,
                                n_components=int(n_components),
                                scale_numeric=scale_numeric,
                            )
                        except Exception as error:
                            st.error(f"Could not run PCA: {error}")
                        else:
                            save_pca_result_to_session(st.session_state, pca_result)
                            st.success("PCA completed. The working data was not changed.")

                    pca_result = st.session_state.get("latest_pca_result")
                    if pca_result:
                        _show_pca_result(pca_result, working_df, color_column)

                        st.subheader("Add PCA Scores")
                        st.caption("This is optional. PCA scores are added to working_df only after you confirm.")
                        score_prefix = st.text_input(
                            "PC score column prefix",
                            value="PC",
                            key="pca_score_prefix",
                            help="Prefix for optional PCA score columns added to working_df.",
                        )
                        if st.button(
                            "Add PC scores to working data",
                            help="Add PCA score columns to working_df only after confirmation.",
                        ):
                            try:
                                new_df, log_entry = add_pca_scores_to_dataframe(
                                    working_df,
                                    pca_result,
                                    score_prefix=score_prefix.strip() or "PC",
                                )
                            except Exception as error:
                                st.error(f"Could not add PCA scores: {error}")
                            else:
                                apply_transformation_result(new_df, log_entry)
                                st.success("PCA score columns were added to the working dataset.")

    elif task_type == "Clustering":
        numeric_feature_options = [
            column
            for column in working_df.columns
            if pd.api.types.is_numeric_dtype(working_df[column])
        ]

        st.info(CLUSTERING_NOTE)

        if len(numeric_feature_options) < 2 or len(working_df) < 2:
            st.warning("Clustering needs at least two numeric features and at least two rows.")
        else:
            feature_columns = st.multiselect(
                "Numeric features",
                numeric_feature_options,
                default=numeric_feature_options[: min(4, len(numeric_feature_options))],
                key="clustering_feature_columns",
                help="Choose numeric variables used to form clusters.",
            )
            if len(feature_columns) < 2:
                st.warning("Choose at least two numeric features.")
            else:
                method = st.selectbox(
                    "Clustering method",
                    ["KMeans", "Agglomerative Clustering", "DBSCAN"],
                    key="clustering_method",
                    help="Choose the clustering algorithm. Clusters are exploratory groupings, not confirmed real-world categories.",
                )
                n_clusters = 3
                dbscan_eps = 0.5
                dbscan_min_samples = 5
                linkage = "ward"
                compute_elbow = False

                if method in {"KMeans", "Agglomerative Clustering"}:
                    max_clusters = min(10, len(working_df))
                    n_clusters = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=max_clusters,
                        value=min(3, max_clusters),
                        step=1,
                        key="clustering_n_clusters",
                        help="Number of groups to form for KMeans or Agglomerative Clustering.",
                    )
                if method == "Agglomerative Clustering":
                    linkage = st.selectbox(
                        "Linkage",
                        ["ward", "complete", "average", "single"],
                        key="clustering_linkage",
                        help="Rule for measuring distances between clusters during agglomerative clustering.",
                    )
                if method == "DBSCAN":
                    dbscan_eps = st.number_input(
                        "DBSCAN eps",
                        min_value=0.01,
                        value=0.5,
                        step=0.05,
                        key="clustering_dbscan_eps",
                        help="Neighborhood radius for DBSCAN. Larger values usually create fewer noise points.",
                    )
                    dbscan_min_samples = st.number_input(
                        "DBSCAN min_samples",
                        min_value=1,
                        value=5,
                        step=1,
                        key="clustering_dbscan_min_samples",
                        help="Minimum nearby rows required for DBSCAN to form a dense region.",
                    )
                if method == "KMeans":
                    compute_elbow = st.checkbox(
                        "Show KMeans elbow plot",
                        value=True,
                        key="clustering_compute_elbow",
                        help="Show inertia across k values as a rough guide for choosing cluster count.",
                    )

                if st.button(
                    "Run clustering",
                    help="Run clustering on the selected features. The working dataset is not changed unless you later add labels.",
                ):
                    try:
                        clustering_result = run_clustering_analysis(
                            working_df,
                            feature_columns=feature_columns,
                            method=method,
                            scale_numeric=scale_numeric,
                            n_clusters=int(n_clusters),
                            random_state=42,
                            dbscan_eps=float(dbscan_eps),
                            dbscan_min_samples=int(dbscan_min_samples),
                            linkage=linkage,
                        )
                        if method == "KMeans" and compute_elbow:
                            clustering_result["elbow_table"] = compute_kmeans_elbow_table(
                                working_df,
                                feature_columns=feature_columns,
                                scale_numeric=scale_numeric,
                                max_k=min(10, len(working_df)),
                                random_state=42,
                            )
                    except Exception as error:
                        st.error(f"Could not run clustering: {error}")
                    else:
                        save_clustering_result_to_session(st.session_state, clustering_result)
                        st.success("Clustering completed. The working data was not changed.")

                clustering_result = st.session_state.get("latest_clustering_result")
                if clustering_result:
                    _show_clustering_result(clustering_result)

                    st.subheader("Add Cluster Labels")
                    st.caption("This is optional. Cluster labels are added to working_df only after you confirm.")
                    new_column = st.text_input(
                        "Cluster label column name",
                        value="cluster_label",
                        key="clustering_label_column",
                        help="Name for the optional cluster label column added to working_df.",
                    )
                    if st.button(
                        "Add cluster labels to working data",
                        help="Add cluster labels to working_df only after confirmation.",
                    ):
                        try:
                            new_df, log_entry = add_cluster_labels_to_dataframe(
                                working_df,
                                clustering_result,
                                new_column=new_column.strip() or "cluster_label",
                            )
                        except Exception as error:
                            st.error(f"Could not add cluster labels: {error}")
                        else:
                            apply_transformation_result(new_df, log_entry)
                            st.success("Cluster labels were added to the working dataset.")

    else:
        numeric_feature_options = [
            column
            for column in working_df.columns
            if pd.api.types.is_numeric_dtype(working_df[column])
        ]

        st.warning(ANOMALY_WARNING)

        if len(numeric_feature_options) < 1 or len(working_df) < 2:
            st.warning("Anomaly detection needs at least one numeric feature and at least two rows.")
        else:
            feature_columns = st.multiselect(
                "Numeric features",
                numeric_feature_options,
                default=numeric_feature_options[: min(4, len(numeric_feature_options))],
                key="anomaly_feature_columns",
                help="Choose numeric variables used to flag unusual rows.",
            )
            if not feature_columns:
                st.warning("Choose at least one numeric feature.")
            else:
                method = st.selectbox(
                    "Anomaly detection method",
                    ["IQR rule", "Z-score", "Isolation Forest", "Local Outlier Factor"],
                    key="anomaly_method",
                    help="Choose the method used to flag unusual rows. Flags are exploratory and do not prove errors.",
                )
                iqr_multiplier = 1.5
                z_threshold = 3.0
                contamination = 0.1
                n_neighbors = 20

                if method == "IQR rule":
                    iqr_multiplier = st.number_input(
                        "IQR multiplier",
                        min_value=0.1,
                        value=1.5,
                        step=0.1,
                        key="anomaly_iqr_multiplier",
                        help="Larger values flag fewer IQR outliers.",
                    )
                elif method == "Z-score":
                    z_threshold = st.number_input(
                        "Z-score threshold",
                        min_value=0.1,
                        value=3.0,
                        step=0.1,
                        key="anomaly_z_threshold",
                        help="Rows farther than this many standard deviations from the mean are flagged.",
                    )
                else:
                    contamination = st.slider(
                        "Expected anomaly fraction",
                        min_value=0.01,
                        max_value=0.49,
                        value=0.10,
                        step=0.01,
                        key="anomaly_contamination",
                        help="Approximate fraction of rows expected to be unusual.",
                    )
                    if method == "Local Outlier Factor":
                        n_neighbors = st.number_input(
                            "LOF neighbors",
                            min_value=1,
                            max_value=max(1, len(working_df) - 1),
                            value=min(20, max(1, len(working_df) - 1)),
                            step=1,
                            key="anomaly_lof_neighbors",
                            help="Number of nearby rows used by Local Outlier Factor.",
                        )

                top_n = st.slider(
                    "Top anomalous rows to display",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    key="anomaly_top_n",
                    help="Limit how many of the most unusual rows are shown in the results table.",
                )

                if st.button(
                    "Run anomaly detection",
                    help="Flag unusual rows without removing or changing data.",
                ):
                    try:
                        anomaly_result = run_anomaly_detection(
                            working_df,
                            feature_columns=feature_columns,
                            method=method,
                            scale_numeric=scale_numeric,
                            iqr_multiplier=float(iqr_multiplier),
                            z_threshold=float(z_threshold),
                            contamination=float(contamination),
                            n_neighbors=int(n_neighbors),
                            random_state=42,
                            top_n=int(top_n),
                        )
                    except Exception as error:
                        st.error(f"Could not run anomaly detection: {error}")
                    else:
                        save_anomaly_result_to_session(st.session_state, anomaly_result)
                        st.success("Anomaly detection completed. No rows were removed and the working data was not changed.")

                anomaly_result = st.session_state.get("latest_anomaly_result")
                if anomaly_result:
                    _show_anomaly_result(anomaly_result)

                    st.subheader("Add Anomaly Flag")
                    st.caption("This is optional. Anomaly flags are added to working_df only after you confirm.")
                    new_column = st.text_input(
                        "Anomaly flag column name",
                        value="anomaly_flag",
                        key="anomaly_flag_column",
                        help="Name for the optional True/False anomaly flag column added to working_df.",
                    )
                    if st.button(
                        "Add anomaly flag to working data",
                        help="Add anomaly flags to working_df only after confirmation.",
                    ):
                        try:
                            new_df, log_entry = add_anomaly_flag_to_dataframe(
                                working_df,
                                anomaly_result,
                                new_column=new_column.strip() or "anomaly_flag",
                            )
                        except Exception as error:
                            st.error(f"Could not add anomaly flag: {error}")
                        else:
                            apply_transformation_result(new_df, log_entry)
                            st.success("Anomaly flag column was added to the working dataset.")

    st.divider()
    _render_saved_model_interpretation(working_df)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import build_preprocessing_pipeline


def _dense(values):
    """Convert sparse or dense sklearn output into a NumPy array for tests."""
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def test_build_preprocessing_pipeline_returns_column_transformer_or_pipeline():
    df = pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["A", "B", "A"],
        }
    )

    preprocessing = build_preprocessing_pipeline(df, ["age", "city"])

    assert isinstance(preprocessing, (ColumnTransformer, Pipeline))


def test_build_preprocessing_pipeline_does_not_fit_immediately():
    df = pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["A", "B", "A"],
        }
    )

    preprocessing = build_preprocessing_pipeline(df, ["age", "city"])

    assert not hasattr(preprocessing, "transformers_")


def test_preprocessing_can_fit_transform_training_data():
    train_df = pd.DataFrame(
        {
            "age": [21, np.nan, 42, 29],
            "income": [40000, 52000, np.nan, 61000],
            "city": ["A", "B", "A", np.nan],
        }
    )
    preprocessing = build_preprocessing_pipeline(train_df, ["age", "income", "city"])

    transformed = _dense(preprocessing.fit_transform(train_df))

    assert transformed.shape[0] == len(train_df)
    assert not np.isnan(transformed).any()


def test_preprocessing_can_transform_test_data():
    train_df = pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["A", "B", "A"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [50, np.nan],
            "city": ["B", "A"],
        }
    )
    preprocessing = build_preprocessing_pipeline(train_df, ["age", "city"])

    train_transformed = _dense(preprocessing.fit_transform(train_df))
    test_transformed = _dense(preprocessing.transform(test_df))

    assert test_transformed.shape[0] == len(test_df)
    assert test_transformed.shape[1] == train_transformed.shape[1]
    assert not np.isnan(test_transformed).any()


def test_unknown_categorical_values_do_not_crash_transformation():
    train_df = pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["A", "B", "A"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [28],
            "city": ["New unseen city"],
        }
    )
    preprocessing = build_preprocessing_pipeline(train_df, ["age", "city"])

    preprocessing.fit(train_df)
    transformed = _dense(preprocessing.transform(test_df))

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_numeric_missing_values_are_handled():
    train_df = pd.DataFrame(
        {
            "age": [21, np.nan, 42],
            "income": [40000, 52000, np.nan],
        }
    )
    preprocessing = build_preprocessing_pipeline(train_df, ["age", "income"])

    transformed = _dense(preprocessing.fit_transform(train_df))

    assert not np.isnan(transformed).any()


def test_categorical_missing_values_are_handled():
    train_df = pd.DataFrame(
        {
            "city": ["A", np.nan, "A", "B"],
            "segment": ["retail", "business", np.nan, "retail"],
        }
    )
    preprocessing = build_preprocessing_pipeline(train_df, ["city", "segment"])

    transformed = _dense(preprocessing.fit_transform(train_df))

    assert transformed.shape[0] == len(train_df)
    assert not np.isnan(transformed).any()


def test_scale_numeric_adds_standard_scaler_step():
    df = pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["A", "B", "A"],
        }
    )

    preprocessing = build_preprocessing_pipeline(df, ["age", "city"], scale_numeric=True)
    numeric_pipeline = {
        name: transformer
        for name, transformer, _columns in preprocessing.transformers
    }["numeric"]

    assert "scaler" in numeric_pipeline.named_steps

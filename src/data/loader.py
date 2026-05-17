"""Functions for loading user-uploaded data files."""

from typing import BinaryIO

import pandas as pd


def load_uploaded_file(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load a Streamlit-uploaded CSV or Excel file into a DataFrame."""
    name = getattr(uploaded_file, "name", "").lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel data.")

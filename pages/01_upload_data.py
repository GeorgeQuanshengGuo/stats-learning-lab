"""Streamlit page for uploading CSV and Excel data."""

import streamlit as st

from src.core.state import set_uploaded_data
from src.data.loader import load_uploaded_file
from src.ui.page_templates import render_educational_page_header
from src.ui.simple_sidebar import render_simple_sidebar
from src.ui.style_loader import load_global_styles


load_global_styles()
render_simple_sidebar()
render_educational_page_header(
    title="Upload Data",
    purpose="Upload a CSV or Excel file and create safe original and working dataset copies.",
    when_to_use="Use this page at the beginning of an analysis or whenever you want to start over with a new file.",
    common_mistake="Do not treat the working dataset as the untouched source. The original dataset is kept separately.",
    key_terms=[("original_dataset", "Original dataset"), ("working_dataset", "Working dataset")],
    next_step="After uploading, inspect the dataset on the EDA page.",
    next_page="pages/02_eda.py",
    tags=["Start", "Data upload"],
)

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload the raw dataset. The app will keep an untouched original copy and create a separate working copy for analysis.",
)

if uploaded_file is not None:
    try:
        uploaded_df = load_uploaded_file(uploaded_file)
    except ValueError as error:
        st.error(str(error))
    else:
        # Keep two separate copies:
        # - original_df is the untouched upload.
        # - working_df is the copy future cleaning steps may change.
        set_uploaded_data(uploaded_df)
        st.session_state["uploaded_file_name"] = uploaded_file.name

        st.success("File uploaded successfully.")
        st.write("Preview of the working dataset:")
        st.dataframe(st.session_state["working_df"].head(50), use_container_width=True)

        st.caption(
            "The original dataset is stored separately and should not be modified "
            "by future cleaning operations."
        )

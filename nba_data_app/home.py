from pathlib import Path
import streamlit as st
import polars as pl
from utils._utils import (
    load_data, 
    get_duckdb_connection
)

### App Layout

st.title("Predicting NBA Careers Dashboard")
st.markdown("---")
st.markdown(
    """

    Authors:

    - Rasheed Martin
    - Justin Cox
    - Juan Sanchez
    - Matthew Wissell

    This project aims to analyze basketball player data and build predictive models to forecast NBA career outcomes. Using a variety of data science techniques, we explore player performance metrics, historical data, and other relevant features to understand factors that influence career trajectories in professional basketball.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("[📂 View this project on GitHub](https://github.com/RasheedMartin/PredictingNBACareers)")


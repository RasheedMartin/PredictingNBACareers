"""Streamlit entrypoint:  streamlit run nba_data_app/app.py"""
import sys
from pathlib import Path

# Make the repo-root `nba_careers` package importable. Appended (not prepended) so
# `utils` keeps resolving to nba_data_app/utils rather than the repo-root utils/.
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Predicting NBA Careers", page_icon="🏀", layout="wide")

pages = {
    "Overview": [st.Page("home.py", title="Home", icon=":material/home:", default=True)],
    "Predictions": [
        st.Page("views/player_lookup.py", title="Player Lookup", icon=":material/person_search:"),
        st.Page("views/rookie_forecasts.py", title="Rookie Forecasts", icon=":material/trending_up:"),
    ],
    "Research": [
        st.Page("views/model_performance.py", title="Model & Methodology", icon=":material/insights:"),
        st.Page("views/EDA_NBA_Draft_Combine.py", title="Draft Combine EDA", icon=":material/query_stats:"),
    ],
}
st.navigation(pages).run()

"""Cached artifact loaders and shared display helpers for the Streamlit app."""
import json

import joblib
import pandas as pd
import streamlit as st

from nba_careers import config

TIER_COLORS = {"Bust Player": "#c8102e", "Role Player": "#f2a900", "Star Player": "#1d428a"}
TIER_SHORT = {"Bust Player": "Bust", "Role Player": "Role", "Star Player": "Star"}
PROB_COLS = {"Bust Player": "p_bust", "Role Player": "p_role", "Star Player": "p_star"}

STAT_LABELS = {
    "Player_Age": "Age", "G": "Games", "MP": "Minutes", "PTS": "Points", "TRB": "Rebounds",
    "AST": "Assists", "PER": "PER", "WS": "Win Shares", "BPM": "Box +/-", "VORP": "VORP",
}


def season_label(start_year) -> str:
    y = int(start_year)
    return f"{y}-{(y + 1) % 100:02d}"


@st.cache_data
def load_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text())


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(config.PREDICTIONS_PATH)
    df["Rookie Season"] = df["Season"].map(season_label)
    df["confidence"] = df[list(PROB_COLS.values())].max(axis=1)
    return df


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(config.IMPORTANCE_PATH)


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)


def pretty_feature(name: str) -> str:
    """'PTS_third' -> 'Points (year 3)', 'AST_growth_y1_y2' -> 'Assists growth (y1->y2)'."""
    for suffix, tag in [("_avg_yoy_growth", " avg. growth"), ("_growth_y1_y2", " growth (y1→y2)"),
                        ("_growth_y2_y3", " growth (y2→y3)"), ("_third", " (year 3)")]:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return STAT_LABELS.get(base, base) + tag
    if name.startswith(("ROY_", "All_Rookie_")) or name == "team_made_playoffs_last_year":
        return name.replace("_", " ")
    return STAT_LABELS.get(name, name) + " (rookie)"


def footer():
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Master's senior design project · Rasheed Martin, Justin Cox, Juan Sanchez, Matthew Wissell"
    )
    st.sidebar.markdown("[View the code on GitHub](https://github.com/RasheedMartin/PredictingNBACareers)")

"""Feature definitions and stateless cleaning steps."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns used to *build* the label (career aggregates). Using any of them as a
# feature would leak the answer, so they are always excluded.
LABEL_SOURCE_COLUMNS = [
    "All_Nba_Team_Max",
    "AllStar_Max",
    "Games_Played_Mean",
    "Win_Share_Sum",
    "Box_Plus_Minus_Mean",
    "Minutes_Played_Mean",
    "Games_Started_Mean",
    "Usage_Mean",
    "Player_Score",
    "Player_Label",
]

# Identifiers / bookkeeping, not predictive signal.
ID_COLUMNS = ["Player", "bbref_id", "left_censored", "Season", "Season_third", "Team", "Draft_Class",
              "All_Rookie_Rank"]

# Draft position only exists for 2000+ rookies in this dataset; before that it is
# always missing, so it would act as an "era" flag rather than a scouting signal.
ERA_CONFOUNDED_COLUMNS = ["Draft_Pick"]

EXCLUDED = set(LABEL_SOURCE_COLUMNS + ID_COLUMNS + ERA_CONFOUNDED_COLUMNS)

# Stats surfaced in the app next to each prediction.
DISPLAY_STATS = [
    "Player_Age", "G", "MP", "PTS", "TRB", "AST", "PER", "WS", "BPM", "VORP",
    "G_third", "MP_third", "PTS_third", "TRB_third", "AST_third", "PER_third",
    "WS_third", "BPM_third", "VORP_third",
]


def candidate_features(df: pd.DataFrame) -> list[str]:
    """All numeric columns that are legal model inputs: rookie-year stats,
    third-season stats, year-over-year growth, and rookie award voting."""
    return [
        c for c in df.columns
        if c not in EXCLUDED and pd.api.types.is_numeric_dtype(df[c])
    ]


def _to_float_frame(X) -> np.ndarray:
    arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    return arr.copy()


class FiniteCleaner(BaseEstimator, TransformerMixin):
    """Replace +/-inf (from growth ratios that divide by zero) with NaN so the
    imputer handles them consistently."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        arr = _to_float_frame(X)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)]))

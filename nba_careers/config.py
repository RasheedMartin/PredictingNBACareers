"""Paths and constants shared by the training pipeline and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELING_TABLE = (
    ROOT / "data/refined_raw_data/Player_Type_Modeling/player_3tier_label_with_target_all_years.csv"
)
APP_DB = ROOT / "data/app/nba_data.duckdb"

ARTIFACT_DIR = ROOT / "models"
MODEL_PATH = ARTIFACT_DIR / "career_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.parquet"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"

TARGET = "Player_Label"
LABELS = ["Bust Player", "Role Player", "Star Player"]

# Rookie seasons up to this year have had time for a career to develop (5+ seasons
# of data through 2024). Later rookie classes are scored as forecasts, not training data.
LAST_RESOLVED_ROOKIE_SEASON = 2019
# Temporal holdout: the most recent resolved rookie classes are never seen in tuning.
HOLDOUT_START_SEASON = 2015

RANDOM_STATE = 42

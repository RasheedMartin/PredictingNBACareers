"""Load the modeling table and split it into honest train / holdout / forecast sets."""
from dataclasses import dataclass

import pandas as pd
import polars as pl

from . import config


@dataclass
class Splits:
    full: pd.DataFrame       # every row, annotated with `split`
    train: pd.DataFrame      # resolved careers, rookie season < HOLDOUT_START_SEASON
    holdout: pd.DataFrame    # resolved careers, most recent rookie classes
    forecast: pd.DataFrame   # careers still in progress (scored, never trained on)

    @property
    def resolved(self) -> pd.DataFrame:
        return pd.concat([self.train, self.holdout])


def load_table(path=config.MODELING_TABLE) -> pd.DataFrame:
    df = pl.read_csv(path, infer_schema_length=10_000).to_pandas()
    df["seasons_to_third"] = df["Season_third"] - df["Season"]
    return df


def assign_split(df: pd.DataFrame) -> pd.Series:
    """
    A row is only usable when the player has a *real* third season
    (Season_third == Season + 2). For shorter careers the source data fills
    "third season" with the last season played, and the career-average label is
    computed from those same seasons, so features and label overlap (leakage).
    """
    real_third = df["seasons_to_third"] == 2
    resolved = df["Season"] <= config.LAST_RESOLVED_ROOKIE_SEASON
    holdout = df["Season"] >= config.HOLDOUT_START_SEASON

    split = pd.Series("excluded", index=df.index)
    split[real_third & resolved & ~holdout] = "train"
    split[real_third & resolved & holdout] = "holdout"
    split[real_third & ~resolved] = "forecast"
    return split


def load_splits(path=config.MODELING_TABLE) -> Splits:
    df = load_table(path)
    df["split"] = assign_split(df)
    return Splits(
        full=df,
        train=df[df.split == "train"].copy(),
        holdout=df[df.split == "holdout"].copy(),
        forecast=df[df.split == "forecast"].copy(),
    )


def encode_target(labels: pd.Series):
    return labels.map({name: i for i, name in enumerate(config.LABELS)}).to_numpy()

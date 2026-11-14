import json

import joblib
import numpy as np
import pandas as pd
import pytest

from nba_careers import config
from nba_careers.data import encode_target, load_splits
from nba_careers.features import LABEL_SOURCE_COLUMNS, candidate_features
from nba_careers.pipeline import build_pipeline


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_splits_only_use_real_third_seasons(splits):
    for frame in (splits.train, splits.holdout, splits.forecast):
        assert (frame["seasons_to_third"] == 2).all()


def test_splits_are_temporally_ordered_and_disjoint(splits):
    assert splits.train.Season.max() < config.HOLDOUT_START_SEASON <= splits.holdout.Season.min()
    assert splits.holdout.Season.max() <= config.LAST_RESOLVED_ROOKIE_SEASON < splits.forecast.Season.min()
    assert set(splits.train.index).isdisjoint(splits.holdout.index)


def test_label_source_columns_never_used_as_features(splits):
    features = candidate_features(splits.train)
    assert not set(features) & set(LABEL_SOURCE_COLUMNS)
    assert "Draft_Pick" not in features


def test_pipeline_handles_inf_and_nan(splits):
    features = candidate_features(splits.train)
    sample = splits.train.sample(300, random_state=0).copy()
    sample.loc[sample.index[:5], features[0]] = np.inf
    sample.loc[sample.index[5:10], features[1]] = np.nan
    model = build_pipeline(features, k=10).fit(sample, encode_target(sample[config.TARGET]))
    proba = model.predict_proba(sample)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m nba_careers.train` first")
def test_saved_artifacts_are_consistent(splits):
    model = joblib.load(config.MODEL_PATH)
    assert model.predict_proba(splits.forecast).shape == (len(splits.forecast), 3)

    metrics = json.loads(config.METRICS_PATH.read_text())
    assert metrics["holdout"]["accuracy"] > metrics["baseline_majority_holdout"]["accuracy"]

    preds = pd.read_parquet(config.PREDICTIONS_PATH)
    assert set(preds.split) == {"train (out-of-fold)", "holdout", "forecast"}
    assert preds.loc[preds.split == "forecast", config.TARGET].isna().all()
    prob_cols = ["p_bust", "p_role", "p_star"]
    assert np.allclose(preds[prob_cols].sum(axis=1), 1)

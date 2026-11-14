"""
The multi-stage model pipeline.

Every stage lives inside one sklearn Pipeline, so cross-validation refits all
of it on each training fold, feature selection included. (In the original
notebooks selection ran on the full dataset before splitting, which leaks
test information into the chosen features.)

    raw columns
      -> FiniteCleaner        inf -> NaN
      -> SimpleImputer        median fill
      -> VarianceThreshold    drop constant columns
      -> StandardScaler
      -> SelectKBest          keep the k most informative features (k is tuned)
      -> classifier           LR / Random Forest / Gradient Boosting (tuned)
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import FiniteCleaner


def build_pipeline(features: list[str], estimator=None, k="all") -> Pipeline:
    preprocess = ColumnTransformer(
        [("num", Pipeline([
            ("clean", FiniteCleaner()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
        ]), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", estimator if estimator is not None else LogisticRegression(max_iter=5000)),
    ])


def _k_grid(n_features: int, ks=(15, 30, 60)) -> list:
    return [k for k in ks if k < n_features] + ["all"]


def candidate_searches(n_features: int) -> dict[str, tuple[object, dict]]:
    """Model families and hyper-parameter grids searched during training."""
    k = _k_grid(n_features)
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"select__k": k, "model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
            {"select__k": k, "model__max_depth": [6, 12, None], "model__min_samples_leaf": [1, 4]},
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=False),
            {"select__k": k, "model__learning_rate": [0.03, 0.1], "model__max_depth": [3, None],
             "model__max_iter": [200]},
        ),
    }

"""
End-to-end training run.

    python -m nba_careers.train

Stages:
  1. Load data and split it: train / temporal holdout / forecast (leakage-safe)
  2. Tune each model family with stratified 5-fold CV on the training set
  3. Select the family by CV macro-F1 with the one-standard-error rule
     (the holdout is never used to choose)
  4. Score every family once on the temporal holdout, next to baselines
  5. Permutation feature importance on the holdout
  6. Refit the chosen config on all resolved careers and export artifacts:
       models/career_model.joblib   final fitted pipeline
       models/metrics.json          everything the app's Model page shows
       models/predictions.parquet   out-of-fold / holdout / forecast predictions
       models/feature_importance.csv
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from . import config
from .data import encode_target, load_splits
from .features import DISPLAY_STATS, candidate_features
from .pipeline import build_pipeline, candidate_searches

SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy",
           "balanced_accuracy": "balanced_accuracy", "neg_log_loss": "neg_log_loss"}


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)


def _scores(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, proba, labels=range(len(config.LABELS))),
    }


def _naive_setup_accuracy(full: pd.DataFrame, features: list[str]) -> float:
    """Reproduce the original notebook setup (every row, incl. <3-season careers,
    random CV) to quantify how much the leakage inflated accuracy."""
    rows = full[full[config.TARGET].notna()]
    model = build_pipeline(features, LogisticRegression(max_iter=5000))
    return float(cross_val_score(model, rows, encode_target(rows[config.TARGET]), cv=_cv()).mean())


def run(quick: bool = False) -> dict:
    t0 = time.time()
    splits = load_splits()
    features = candidate_features(splits.train)
    X_tr, y_tr = splits.train, encode_target(splits.train[config.TARGET])
    X_ho, y_ho = splits.holdout, encode_target(splits.holdout[config.TARGET])
    print(f"[data] train={len(X_tr)} holdout={len(X_ho)} forecast={len(splits.forecast)} "
          f"features={len(features)}")

    # ---- Stage 2: tune each family ------------------------------------------------
    searches = candidate_searches(len(features))
    if quick:
        searches = {k: (est, {p: v[:1] for p, v in grid.items()}) for k, (est, grid) in searches.items()}
    families = {}
    for name, (estimator, grid) in searches.items():
        gs = GridSearchCV(build_pipeline(features, estimator), grid, cv=_cv(),
                          scoring=SCORING, refit="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        cv = {m: float(gs.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}
        cv["log_loss"] = -cv.pop("neg_log_loss")
        cv["f1_macro_std"] = float(gs.cv_results_["std_test_f1_macro"][i])
        holdout = _scores(y_ho, gs.best_estimator_.predict_proba(X_ho))
        families[name] = {"best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                                          for k, v in gs.best_params_.items()},
                          "cv": cv, "holdout": holdout, "estimator": gs.best_estimator_}
        print(f"[tune] {name:20s} cv_f1={cv['f1_macro']:.3f} holdout_f1={holdout['f1_macro']:.3f} "
              f"params={gs.best_params_}")

    # ---- Stage 3: choose by CV only (one-standard-error rule) ---------------------
    # Families are listed simplest-first. Pick the simplest one whose CV macro-F1 is
    # within one standard error of the top score; small CV gaps are noise.
    top = max(families, key=lambda n: families[n]["cv"]["f1_macro"])
    se = families[top]["cv"]["f1_macro_std"] / np.sqrt(_cv().get_n_splits())
    threshold = families[top]["cv"]["f1_macro"] - se
    best_name = next(n for n in families if families[n]["cv"]["f1_macro"] >= threshold)
    best = families[best_name]["estimator"]
    print(f"[select] {best_name} (top CV: {top}, 1-SE threshold {threshold:.3f})")

    # ---- Stage 4: holdout report + baselines --------------------------------------
    ho_proba = best.predict_proba(X_ho)
    ho_pred = ho_proba.argmax(axis=1)
    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    baseline = _scores(y_ho, majority.predict_proba(X_ho))
    naive_acc = _naive_setup_accuracy(splits.full, features)

    # ---- Stage 5: permutation importance on the holdout ---------------------------
    pi = permutation_importance(best, X_ho, y_ho, scoring="f1_macro", n_repeats=3 if quick else 10,
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": X_ho.columns, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                  .query("feature in @features")
                  .sort_values("importance", ascending=False))

    # ---- Stage 6: final refit + predictions ---------------------------------------
    resolved = splits.resolved
    y_res = encode_target(resolved[config.TARGET])
    final = clone(best).fit(resolved, y_res)

    # Predictions shown in the app are never in-sample: training players get
    # out-of-fold predictions, holdout players get holdout predictions, and
    # in-progress careers get the final model's forecast.
    oof = cross_val_predict(clone(best), X_tr, y_tr, cv=_cv(), method="predict_proba")
    fc = final.predict_proba(splits.forecast)
    pred_frames = []
    for frame, proba, source in [(X_tr, oof, "train (out-of-fold)"),
                                 (X_ho, ho_proba, "holdout"),
                                 (splits.forecast, fc, "forecast")]:
        out = frame[["Player", "Season", "Draft_Pick", "Team", config.TARGET, *DISPLAY_STATS]].copy()
        out["split"] = source
        for i, label in enumerate(config.LABELS):
            out[f"p_{label.split()[0].lower()}"] = proba[:, i]
        out["predicted"] = [config.LABELS[i] for i in proba.argmax(axis=1)]
        pred_frames.append(out)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.loc[predictions.split == "forecast", config.TARGET] = None  # careers unresolved

    selected_mask = final.named_steps["select"].get_support()
    selected_names = final.named_steps["preprocess"].get_feature_names_out()[selected_mask].tolist()

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "labels": config.LABELS,
        "data": {
            "source": str(config.MODELING_TABLE.relative_to(config.ROOT)),
            "rows_total": int(len(splits.full)),
            "rows_excluded_short_career": int((splits.full.split == "excluded").sum()),
            "train": {"n": int(len(X_tr)), "seasons": [int(X_tr.Season.min()), int(X_tr.Season.max())],
                      "label_counts": X_tr[config.TARGET].value_counts().to_dict()},
            "holdout": {"n": int(len(X_ho)), "seasons": [int(X_ho.Season.min()), int(X_ho.Season.max())],
                        "label_counts": X_ho[config.TARGET].value_counts().to_dict()},
            "forecast": {"n": int(len(splits.forecast)),
                         "seasons": [int(splits.forecast.Season.min()), int(splits.forecast.Season.max())]},
            "n_candidate_features": len(features),
        },
        "selected_model": best_name,
        "selection_rule": {"rule": "one-standard-error, simplest family first",
                           "top_cv_family": top, "threshold_f1_macro": threshold},
        "selected_params": families[best_name]["best_params"],
        "selected_features": selected_names,
        "families": {n: {k: v for k, v in f.items() if k != "estimator"} for n, f in families.items()},
        "holdout": {
            **_scores(y_ho, ho_proba),
            "confusion_matrix": confusion_matrix(y_ho, ho_pred, labels=range(3)).tolist(),
            "report": classification_report(y_ho, ho_pred, target_names=config.LABELS, output_dict=True),
        },
        "baseline_majority_holdout": baseline,
        "leakage": {
            "naive_setup_cv_accuracy": naive_acc,
            "note": "Original setup: all rows including players with <3 seasons, random CV.",
        },
        "train_seconds": round(time.time() - t0, 1),
    }

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_PATH, compress=3)
    predictions.to_parquet(config.PREDICTIONS_PATH, index=False)
    importance.to_csv(config.IMPORTANCE_PATH, index=False)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[done] holdout acc={metrics['holdout']['accuracy']:.3f} "
          f"f1={metrics['holdout']['f1_macro']:.3f} (baseline acc={baseline['accuracy']:.3f}, "
          f"naive-setup acc={naive_acc:.3f}) in {metrics['train_seconds']}s")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny grids, for smoke tests")
    run(quick=parser.parse_args().quick)

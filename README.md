# Predicting NBA Careers

> **Can a player's first three NBA seasons tell us whether they'll become a Star, a Role Player, or a Bust?**

Master's senior design project by **Rasheed Martin, Justin Cox, Juan Sanchez, and Matthew Wissell**.

**Live app:** https://predicting-nba-careers-production.up.railway.app

| | |
|---|---|
| Holdout accuracy (unseen 2015–19 rookie classes) | **72.2%** (majority baseline 33.9%) |
| Holdout macro F1 | **0.73** |
| Star ↔ Bust confusions on holdout | **0 of 313** players |
| Careers forecast (2020–23 rookie classes) | 249 |

Three seasons is also when teams decide on rookie-scale contract extensions, so the model answers a real front-office question: *is this player on a starter trajectory?*

## What's in the app

- **Player Lookup:** search any player and get the predicted career tier, tier probabilities, and a stat comparison.
- **Rookie Forecasts:** predictions for careers still in progress (through the 2023-24 rookie class, Wembanyama's year), plus a report card on the held-out 2015–19 classes.
- **Model & Methodology:** the pipeline, model comparison, confusion matrix, feature importance, and the data-leakage fixes.
- **Draft Combine EDA:** interactive exploratory analysis of NBA Draft Combine measurements.

## End-to-end pipeline

```
Basketball-Reference ──fetch──► data/raw_data/*.csv ──warehouse──► DuckDB warehouse ──table──► modeling table ──train──► model + app
      (per season)               (same format as the     (16 tables, keyed by        (1 row/player,       (models/)
                                  original exports)       player id)                  258 columns)
```

```bash
python -m nba_careers.etl refresh      # fetch any new completed seasons, rebuild everything, retrain (~40s + fetch)
python -m nba_careers.etl fetch --season 2026   # or run stages individually: fetch | warehouse | table
python -m nba_careers.train            # retrain only
```

### Getting new data

New seasons are fetched automatically from Basketball-Reference: about 6 requests per season, rate-limited to stay under their 20 requests/minute limit, with every page cached under `data/raw_html/`.

- **Scheduled:** the `Refresh NBA data` GitHub Action runs every July 15, after the awards and the draft. It fetches the new season, rebuilds, retrains, and opens a pull request with the new data and metrics for review. Merging redeploys the app.
- **On demand:** run the Action manually from the Actions tab, or run `python -m nba_careers.etl refresh` locally. If Basketball-Reference blocks GitHub's servers, run it locally and push.
- **Draft Combine** measurements come from a Kaggle dataset and are updated manually (`data/raw_data/NBA_Draft_Combine/`).

The fetcher is validated against the team's original hand-exported CSVs. For 2024-25, the voting tables and team tables match exactly. The remaining differences are corrections Basketball-Reference has made since then: 8 position changes, 1-minute stat fixes, and accented names.

### Data quality fixes in the rebuilt warehouse

Rebuilding the notebook ETL as a tested pipeline exposed issues in the original database:

| Issue in the original | Effect | Fix |
|---|---|---|
| Players matched by name | Fathers and sons merged (Patrick Ewing Sr./Jr., Mike Dunleavy Sr./Jr., Gerald Henderson Sr./Jr.) | Keyed by Basketball-Reference player ID |
| Traded players' per-team rows loaded alongside the combined row | Some seasons counted 2–3× in career totals | One row per player-season, enforced and tested |
| Unmapped team codes (2TM, KCK, ...) | 15% of stat rows assigned to a single fake "Unknown" team | Complete franchise map |
| Data starts in 1976-77 | Veterans active that season treated as rookies | Flagged `left_censored` and excluded from training |
| "Last First" combine names without a comma | 54 players named "None …" | Parser handles both formats |
| Two hand-exported team files had shifted headers | Misaligned columns | Columns derived from the stable header row |

The pipeline is deterministic. CI rebuilds the warehouse and modeling table from the raw CSVs on every push and fails if the result differs from what's committed.

## The ML pipeline

```
modeling table ─► leakage-safe split ─► ┌── sklearn Pipeline (refit in every CV fold) ──┐ ─► artifacts
  3,983 players     train   1545          │ inf→NaN · median impute · drop constants     │    models/career_model.joblib
                    holdout  313          │ standardize · SelectKBest (k tuned)          │    models/metrics.json
                    forecast 249          │ classifier: LR / RF / GBM (tuned)            │    models/predictions.parquet
                                          └───────────────────────────────────────────────┘    models/feature_importance.csv
```

1. **Split.** Only players with a real rookie season and a real third season. Rookie seasons ≤2014 are used for training, 2015–19 are a temporal holdout, and 2020–23 are forecasts (careers still in progress).
2. **Tune.** Logistic Regression, Random Forest, and Gradient Boosting are each grid-searched with stratified 5-fold CV. The number of selected features is tuned alongside each model's hyperparameters.
3. **Select.** The one-standard-error rule picks the simplest model family within one SE of the best CV macro-F1. The holdout is never used to choose.
4. **Evaluate.** The chosen model is scored once on the holdout, with permutation feature importance.
5. **Export.** The model is refit on all resolved careers. Predictions shown in the app are never in-sample: training players get out-of-fold predictions.

### Label leakage we found and fixed

The original notebooks reported ~82% accuracy, which was inflated:

- For players with fewer than three seasons, "third-season" stats were filled with their *last* season, and the career-average label was computed from those same seasons.
- Feature selection (correlation filter, RFECV) ran on the full dataset before the train/test split. It now happens inside the pipeline.
- Random splits mixed eras. Evaluation now uses a temporal holdout.

### Known limitations

- **"Star" means a long, productive starter-level career** (career score > 0.4), not All-Star. About half of resolved players qualify.
- **Probabilities are not calibrated.** Logistic regression is overconfident at the extremes. Use the tier, not the exact percentage.
- **No contract context:** salary, injuries, age curves, and positional value are not modeled.

## Project layout

```
nba_careers/etl/    data pipeline: fetch.py (Basketball-Reference), warehouse.py (DuckDB), modeling_table.py
nba_careers/        training pipeline (config, data, features, pipeline, train)
nba_data_app/       Streamlit app (app.py entrypoint, views/, lib.py)
models/             trained artifacts committed for the app
data/raw_data/      raw season CSVs (source of truth for the warehouse)
data/app/           slim DuckDB (combine tables only) used by the EDA page
data/warehouse/     full DuckDB warehouse (built locally, git-ignored)
notebooks/          original data extraction, EDA, and modeling notebooks
tests/              pytest suite (fetch parser, warehouse invariants, leakage guards, artifacts)
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                                # tests
python -m nba_careers.etl refresh     # rebuild data + retrain
streamlit run nba_data_app/app.py     # http://localhost:8501
```

With Docker (the same image that runs in production):

```bash
docker build -t nba-careers .
docker run -p 8501:8501 nba-careers
```

## Deploying (Railway)

Railway's settings live in code, in `.railway/railway.py` (Dockerfile build, health check on `/_stcore/health`). The app honours `$PORT`.

```bash
pip install -r .railway/requirements.txt   # Railway's Python SDK, used by the CLI to read railway.py
railway config plan                         # preview config changes
railway config apply                        # apply them
railway up --service predicting-nba-careers # build & deploy from the local directory
```

Or connect the GitHub repo in the Railway dashboard to deploy automatically on every push to `main`.

`scikit-learn` is pinned in `requirements.txt` because the saved model must be loaded by the version that trained it. Retrain after upgrading.

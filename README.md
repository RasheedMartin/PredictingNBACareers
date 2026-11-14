# Predicting NBA Careers

> **Can a player's first three NBA seasons tell us whether they'll become a Star, a Role Player, or a Bust?**

Master's senior design project by **Rasheed Martin, Justin Cox, Juan Sanchez, and Matthew Wissell**.

**Live app:** _add your Railway URL here_

| | |
|---|---|
| Holdout accuracy (unseen 2015–19 rookie classes) | **71.7%** (majority baseline 34.1%) |
| Holdout macro F1 | **0.72** |
| Star ↔ Bust confusions on holdout | **1 of 314** players |
| Careers forecast (2020–22 rookies) | 188 |

## What's in the app

- **Player Lookup:** search any player and get the predicted career tier, tier probabilities, and a stat comparison.
- **Rookie Forecasts:** live predictions for the 2020–22 rookie classes, plus a report card on the held-out 2015–19 classes.
- **Model & Methodology:** the pipeline, model comparison, confusion matrix, feature importance, and how we found and fixed data leakage.
- **Draft Combine EDA:** interactive exploratory analysis of NBA Draft Combine measurements.

## The ML pipeline

One command trains, evaluates, and exports everything:

```bash
python -m nba_careers.train
```

```
modeling table ─► leakage-safe split ─► ┌── sklearn Pipeline (refit in every CV fold) ──┐ ─► artifacts
  3,855 players     train   1715          │ inf→NaN · median impute · drop constants     │    models/career_model.joblib
                    holdout  314          │ standardize · SelectKBest (k tuned)          │    models/metrics.json
                    forecast 188          │ classifier: LR / RF / GBM (tuned)            │    models/predictions.parquet
                                          └───────────────────────────────────────────────┘    models/feature_importance.csv
```

1. **Split.** Only players with a real third season. Rookie seasons ≤2014 are used for training, 2015–19 are a temporal holdout, and 2020–22 are forecasts (careers still in progress).
2. **Tune.** Logistic Regression, Random Forest, and Gradient Boosting are each grid-searched with stratified 5-fold CV. The number of selected features is tuned alongside each model's hyperparameters.
3. **Select.** The one-standard-error rule picks the simplest model family within one SE of the best CV macro-F1. The holdout is never used to choose.
4. **Evaluate.** The chosen model is scored once on the holdout, with permutation feature importance.
5. **Export.** The model is refit on all resolved careers. Predictions shown in the app are never in-sample: training players get out-of-fold predictions.

### Data leakage we found and fixed

The original notebooks reported ~81% accuracy. Three problems inflated that number:

- For players with fewer than three seasons, "third-season" stats were filled with their *last* season, and the career-average label was computed from those same seasons. 1,638 such rows are now excluded.
- Feature selection (correlation filter, RFECV) ran on the full dataset before the train/test split. It now happens inside the pipeline.
- Random splits mixed eras. Evaluation now uses a temporal holdout.

## Project layout

```
nba_careers/        training pipeline package (config, data, features, pipeline, train)
nba_data_app/       Streamlit app (app.py entrypoint, views/, lib.py)
models/             trained artifacts committed for the app
data/app/           slim DuckDB (combine tables only) used by the EDA page
data/               raw, refined, and clean datasets
notebooks/          data extraction, EDA, and original modeling notebooks
tests/              pytest suite (leakage guards, pipeline, artifact consistency)
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                                # tests
python -m nba_careers.train           # retrain (~40s); rewrites models/
streamlit run nba_data_app/app.py     # http://localhost:8501
```

With Docker (the same image that runs in production):

```bash
docker build -t nba-careers .
docker run -p 8501:8501 nba-careers
```

## Deploying (Railway)

The repo deploys as-is via `railway.toml` (Dockerfile build, health check on `/_stcore/health`, `$PORT` honoured).

```bash
railway login
railway init        # or: railway link (existing project)
railway up          # build & deploy from the local directory
railway domain      # generate a public URL
```

Or connect the GitHub repo in the Railway dashboard to deploy automatically on every push to `main`.

`scikit-learn` is pinned in `requirements.txt` because the saved model must be loaded by the version that trained it. Retrain after upgrading.

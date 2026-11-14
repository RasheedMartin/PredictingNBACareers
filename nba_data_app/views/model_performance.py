import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import TIER_SHORT, footer, load_importance, load_metrics, pretty_feature

m = load_metrics()
ho, base, d = m["holdout"], m["baseline_majority_holdout"], m["data"]
labels = [TIER_SHORT[x] for x in m["labels"]]

st.title("Model & Methodology")
st.caption(f"Model: **{m['selected_model']}** · trained {m['trained_at'][:10]} · "
           f"scikit-learn {m['versions']['scikit-learn']}")

c = st.columns(4)
c[0].metric("Holdout accuracy", f"{ho['accuracy']:.1%}")
c[1].metric("Holdout macro F1", f"{ho['f1_macro']:.3f}")
c[2].metric("Majority-class baseline", f"{base['accuracy']:.1%}")
c[3].metric("Balanced accuracy", f"{ho['balanced_accuracy']:.1%}")

# ---------------------------------------------------------------- pipeline
st.markdown("---")
st.subheader("The pipeline")
st.markdown(
    "Training is one reproducible command (`python -m nba_careers.train`). Every preprocessing and "
    "feature-selection step sits **inside** a single scikit-learn `Pipeline`, so cross-validation "
    "refits all of it on each fold and no information from validation data leaks into training."
)
st.graphviz_chart(f"""
digraph {{
  rankdir=LR; nodesep=0.3; ranksep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#f3f5f9", color="#1d428a", fontname="Helvetica", fontsize=12];
  edge [color="#1d428a"];
  data [label="Modeling table\\n{d['rows_total']:,} players"];
  split [label="Leakage-safe split\\ntrain {d['train']['n']} · holdout {d['holdout']['n']}\\nforecast {d['forecast']['n']}"];
  subgraph cluster_fold {{
    label="sklearn Pipeline (refit inside every CV fold)"; fontname="Helvetica"; fontsize=11;
    style="rounded,dashed"; color="#1d428a";
    prep [label="Preprocess\\ninf → NaN · median impute\\ndrop constants · standardize"];
    select [label="SelectKBest\\n(k tuned)"];
    model [label="Classifier\\nLR · RF · GBM (tuned)", fillcolor="#e6ecf7"];
    prep -> select -> model;
  }}
  out [label="Artifacts\\nmodel · metrics\\npredictions", fillcolor="#e6ecf7"];
  data -> split -> prep; model -> out;
}}
""")

with st.expander("Stage details"):
    st.markdown(f"""
1. **Split.** Only players with a real third season are used. Rookie seasons
   {d['train']['seasons'][0]}–{d['train']['seasons'][1]} are used for training. The
   {d['holdout']['seasons'][0]}–{d['holdout']['seasons'][1]} classes form a **temporal holdout**.
   {d['forecast']['seasons'][0]}–{d['forecast']['seasons'][1]} rookies have careers still in progress,
   so they are only **forecast**.
2. **Features.** {d['n_candidate_features']} candidate inputs: rookie-year stats, third-season stats,
   year-over-year growth, and Rookie of the Year / All-Rookie voting. Columns used to build the label
   and draft position (missing before 2000, so it would only encode the era) are excluded.
3. **Tuning.** Logistic Regression, Random Forest, and Gradient Boosting are each grid-searched with
   stratified 5-fold CV, tuning the number of selected features along with the model's own hyperparameters.
4. **Selection.** We use the *one-standard-error rule*: pick the simplest model family whose CV macro-F1
   is within one standard error of the best. The holdout is never used to choose.
5. **Evaluation.** The chosen model is scored once on the holdout and then refit on all resolved careers.
""")

# ---------------------------------------------------------------- comparison
st.markdown("---")
st.subheader("Model comparison")
fam = pd.DataFrame([
    {"Model": n, "CV macro F1": f["cv"]["f1_macro"], "CV std": f["cv"]["f1_macro_std"],
     "Holdout accuracy": f["holdout"]["accuracy"], "Holdout macro F1": f["holdout"]["f1_macro"],
     "Holdout log loss": f["holdout"]["log_loss"], "Best params": ", ".join(f"{k.split('__')[1]}={v}" for k, v in f["best_params"].items()),
     "Selected": "✅" if n == m["selected_model"] else ""}
    for n, f in m["families"].items()
])
st.dataframe(fam.style.format({c: "{:.3f}" for c in fam.columns if fam[c].dtype == float}),
             hide_index=True, width="stretch")
st.caption("The three families land within noise of each other in CV, so the simpler, more interpretable "
           "logistic regression is preferred. Gaps of a few points on a 314-player holdout are "
           "within sampling error.")

# ---------------------------------------------------------------- confusion + per class
st.markdown("---")
l, r = st.columns(2, gap="large")
with l:
    st.subheader("Confusion matrix (holdout)")
    cm = ho["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", showscale=False,
                               text=cm, texttemplate="%{text}", textfont={"size": 18}))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=360,
                      yaxis_autorange="reversed", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
with r:
    st.subheader("Per-tier results (holdout)")
    rep = pd.DataFrame({TIER_SHORT[k]: v for k, v in ho["report"].items() if k in m["labels"]}).T
    rep = rep.rename(columns={"f1-score": "F1", "precision": "Precision", "recall": "Recall", "support": "Players"})
    st.dataframe(rep.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}", "Players": "{:.0f}"}),
                 width="stretch")
    star_bust = cm[0][2] + cm[2][0]
    st.markdown(f"The model almost never mixes up the extremes: only **{star_bust}** of {sum(map(sum, cm))} "
                "holdout players were confused between Star and Bust. Nearly all errors are between "
                "**Role Player** and a neighbouring tier, which is expected for the middle tier.")

# ---------------------------------------------------------------- importance
st.markdown("---")
st.subheader("What drives the predictions")
imp = load_importance().head(20).copy()
imp["Feature"] = imp.feature.map(pretty_feature)
fig = px.bar(imp[::-1], x="importance", y="Feature", error_x="std", orientation="h",
             labels={"importance": "Drop in holdout macro-F1 when shuffled", "Feature": ""})
fig.update_traces(marker_color="#1d428a")
fig.update_layout(height=560, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("Permutation importance on the holdout. Third-season role (games started, minutes, usage) "
           "and impact metrics (BPM, win shares) dominate. How a team *uses* a player by year three "
           "is the strongest early signal of career trajectory.")

# ---------------------------------------------------------------- leakage
st.markdown("---")
st.subheader("Catching data leakage")
lk = m["leakage"]["naive_setup_cv_accuracy"]
st.markdown(f"""
The first version of this model scored **{lk:.0%}** in cross-validation. On review we found that number
was inflated:

* **Short careers leaked the label.** For players with fewer than three seasons, the dataset fills
  "third-season" stats with their *last* season, and the career-average label is computed from those same
  seasons. The model was partly reading the answer.
* **Veterans were treated as rookies.** The data starts in 1976-77, so players already in the league
  then had mid-career seasons counted as their "first three". Together with short careers,
  {d['rows_excluded_short_career']:,} such players are now excluded.
* **Fathers and sons were merged.** Players were matched by name, so Patrick Ewing Sr. and Jr. (and others)
  became one "career". The rebuilt data warehouse keys players by Basketball-Reference ID.
* **Feature selection ran before the split.** Correlation filtering and RFECV used every row, including
  test rows. Selection now happens inside the pipeline on each training fold.
* **Random splits mixed eras.** Testing on the most recent rookie classes (temporal holdout) mirrors how
  the model would actually be used.

The honest result is **{ho['accuracy']:.0%}** accuracy on unseen rookie classes, against a
{base['accuracy']:.0%} majority-class baseline. That is lower than the original number, and it's one we
can stand behind.
""")
footer()

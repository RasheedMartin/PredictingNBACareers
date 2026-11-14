import streamlit as st

from lib import footer, load_metrics, load_predictions, year_span

metrics = load_metrics()
preds = load_predictions()
ho = metrics["holdout"]

st.title("Predicting NBA Careers")
st.markdown(
    "#### Can a player's first three seasons tell us whether they'll become a **Star**, "
    "a **Role Player**, or a **Bust**?"
)
st.markdown(
    "We built a machine-learning pipeline on 50 years of NBA data (rookie stats, third-season stats, "
    "year-over-year growth, and rookie award voting) to classify where a career is headed. "
    "The model is evaluated on rookie classes it never saw during training, so the numbers below "
    "reflect how well it predicts the future."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy on unseen rookie classes", f"{ho['accuracy']:.0%}",
          f"+{ho['accuracy'] - metrics['baseline_majority_holdout']['accuracy']:.0%} vs. baseline")
c2.metric("Macro F1", f"{ho['f1_macro']:.2f}")
c3.metric("Players scored", f"{len(preds):,}")
c4.metric(f"Careers forecast ({year_span(metrics['data']['forecast']['seasons'])} rookies)",
          metrics["data"]["forecast"]["n"])

st.markdown("---")
a, b, c = st.columns(3)
with a:
    st.subheader("🔎 Player Lookup")
    st.write("Search any player and see the model's verdict, its confidence, and how they stack up against each tier.")
    st.page_link("views/player_lookup.py", label="Look up a player", icon=":material/arrow_forward:")
with b:
    st.subheader("📈 Rookie Forecasts")
    st.write(f"Who from the {year_span(metrics['data']['forecast']['seasons'])} rookie classes is on a star "
             "trajectory? See every prediction.")
    st.page_link("views/rookie_forecasts.py", label="See the forecasts", icon=":material/arrow_forward:")
with c:
    st.subheader("🧪 Model & Methodology")
    st.write("How the pipeline works, how we caught and fixed data leakage, and full evaluation results.")
    st.page_link("views/model_performance.py", label="Read the methodology", icon=":material/arrow_forward:")

st.markdown("---")
st.subheader("How the career tiers are defined")
st.markdown(
    "Each player gets a **career score** that combines games played, games started, minutes, usage, "
    "win shares, box plus-minus, and All-Star / All-NBA selections. Scores are split into three tiers: "
    "**Bust**, **Role Player**, and **Star Player**. "
    "\"Star\" here means a long, productive starting-calibre career, not only All-Stars."
)

with st.expander("About the project"):
    st.markdown(
        """
        **Authors:** Rasheed Martin, Justin Cox, Juan Sanchez, Matthew Wissell

        This master's senior design project analyzes basketball player data and builds predictive
        models of NBA career outcomes. The project includes data collection (web scraping and public
        datasets), a DuckDB data warehouse, exploratory analysis of draft combine, MVP voting, and team
        data, and a reproducible training pipeline.
        """
    )
footer()

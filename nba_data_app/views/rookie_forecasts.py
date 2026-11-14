import plotly.express as px
import streamlit as st

from lib import TIER_COLORS, footer, load_predictions

preds = load_predictions()

st.title("Rookie Forecasts")

tab_future, tab_report = st.tabs(["Forecasts: 2020–22 rookies", "Report card: 2015–19 rookies"])

TABLE_COLS = {
    "Player": st.column_config.TextColumn("Player"),
    "Rookie Season": st.column_config.TextColumn("Rookie"),
    "Draft_Pick": st.column_config.NumberColumn("Pick", format="%d"),
    "predicted": st.column_config.TextColumn("Predicted"),
    "p_star": st.column_config.ProgressColumn("P(Star)", min_value=0, max_value=1, format="percent"),
    "p_role": st.column_config.ProgressColumn("P(Role)", min_value=0, max_value=1, format="percent"),
    "p_bust": st.column_config.ProgressColumn("P(Bust)", min_value=0, max_value=1, format="percent"),
    "PTS_third": st.column_config.NumberColumn("Yr-3 points", format="%d"),
    "WS_third": st.column_config.NumberColumn("Yr-3 win shares", format="%.1f"),
}

with tab_future:
    st.markdown(
        "These players have three NBA seasons in the books but their careers are still unfolding. "
        "The model has **never seen their outcomes**. Check back in a few years to see how it did."
    )
    fc = preds[preds.split == "forecast"]
    c1, c2 = st.columns([2, 1])
    seasons = c1.multiselect("Rookie season", sorted(fc["Rookie Season"].unique()),
                             default=sorted(fc["Rookie Season"].unique()))
    tiers = c2.multiselect("Predicted tier", list(TIER_COLORS), default=list(TIER_COLORS))
    view = fc[fc["Rookie Season"].isin(seasons) & fc.predicted.isin(tiers)].sort_values("p_star", ascending=False)

    counts = view.predicted.value_counts()
    m = st.columns(3)
    for col, tier in zip(m, ["Star Player", "Role Player", "Bust Player"]):
        col.metric(f"Projected {tier.split()[0]}s", int(counts.get(tier, 0)))

    st.dataframe(view[list(TABLE_COLS)], column_config=TABLE_COLS, hide_index=True,
                 width="stretch", height=420)

    fig = px.scatter(view, x="MP_third", y="p_star", color="predicted", hover_name="Player",
                     color_discrete_map=TIER_COLORS,
                     labels={"MP_third": "Minutes played in year 3", "p_star": "P(Star)", "predicted": ""},
                     title="Year-3 minutes vs. star probability")
    fig.update_layout(yaxis_tickformat=".0%", height=420, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

with tab_report:
    ho = preds[preds.split == "holdout"].copy()
    ho["correct"] = ho.predicted == ho.Player_Label
    st.markdown(
        f"The 2015–19 rookie classes were **held out** of training and model selection entirely. "
        f"The model got **{ho.correct.mean():.0%}** of these {len(ho)} players right."
    )
    acc = (ho.groupby("Rookie Season").correct.mean().rename("Accuracy").reset_index())
    fig = px.bar(acc, x="Rookie Season", y="Accuracy", text_auto=".0%", title="Accuracy by rookie class")
    fig.update_traces(marker_color="#1d428a")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=320)
    st.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    cols = ["Player", "Rookie Season", "Player_Label", "predicted", "confidence"]
    cfg = {"Player_Label": "Actual", "predicted": "Predicted",
           "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent")}
    with l:
        st.subheader("Most confident hits")
        st.dataframe(ho[ho.correct].nlargest(10, "confidence")[cols], column_config=cfg,
                     hide_index=True, width="stretch")
    with r:
        st.subheader("Biggest misses")
        st.dataframe(ho[~ho.correct].nlargest(10, "confidence")[cols], column_config=cfg,
                     hide_index=True, width="stretch")
    st.caption("Misses are often late bloomers or players whose careers were derailed by injury, "
               "which early-career box-score stats can't foresee.")
footer()

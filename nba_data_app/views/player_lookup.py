import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import PROB_COLS, STAT_LABELS, TIER_COLORS, TIER_SHORT, footer, load_predictions

preds = load_predictions()

st.title("Player Lookup")
st.caption("Pick a player to see the model's career-tier prediction from their first three seasons.")

options = (preds.sort_values(["Player", "Season"])
           .assign(key=lambda d: d["Player"] + "  ·  " + d["Rookie Season"])
           .set_index("key"))
keys = options.index.tolist()
default = next((i for i, k in enumerate(keys) if k.startswith("Ja Morant")), 0)
choice = st.selectbox("Player (rookie season)", keys, index=default)
row = options.loc[choice]

SOURCE_NOTE = {
    "forecast": "**Live forecast.** This career is still in progress, so the actual tier isn't known yet.",
    "holdout": "**Holdout player.** This rookie class was hidden from the model during training and tuning.",
    "train (out-of-fold)": "**Out-of-fold prediction.** Made by a model that never saw this player "
                           "(5-fold cross-validation).",
}

left, right = st.columns([1, 1.3], gap="large")
with left:
    pred = row["predicted"]
    st.markdown(
        f"<div style='padding:1.2rem;border-radius:12px;background:{TIER_COLORS[pred]}12;"
        f"border-left:6px solid {TIER_COLORS[pred]}'>"
        f"<div style='font-size:0.9rem;opacity:0.7'>Predicted career tier</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:{TIER_COLORS[pred]}'>{pred}</div>"
        f"<div style='opacity:0.8'>{'>99%' if row['confidence'] > 0.99 else format(row['confidence'], '.0%')}"
        f" confidence · rookie season "
        f"{row['Rookie Season']}</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(SOURCE_NOTE[row["split"]])
    if pd.notna(row["Player_Label"]):
        hit = row["Player_Label"] == pred
        st.markdown(f"Actual career tier: **{row['Player_Label']}** "
                    f"{'✅ correct' if hit else '❌ missed'}")
    meta = []
    if pd.notna(row["Draft_Pick"]):
        meta.append(f"Pick #{int(row['Draft_Pick'])}")
    if pd.notna(row["Team"]):
        meta.append(f"Rookie team: {row['Team']}")
    if meta:
        st.caption(" · ".join(meta))

with right:
    tiers = list(PROB_COLS)
    fig = go.Figure(go.Bar(
        x=[row[PROB_COLS[t]] for t in tiers], y=[TIER_SHORT[t] for t in tiers], orientation="h",
        marker_color=[TIER_COLORS[t] for t in tiers],
        text=[f"{row[PROB_COLS[t]]:.0%}" for t in tiers], textposition="outside",
    ))
    fig.update_layout(title="Probability of each tier", height=260, margin=dict(l=10, r=30, t=40, b=10),
                      xaxis=dict(range=[0, 1.12], tickformat=".0%", showgrid=False),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("How this player compares")
st.caption("Player's stats next to the median player in each actual career tier "
           "(players with resolved careers only).")

resolved = preds[preds["Player_Label"].notna()]
stats = ["G", "MP", "PTS", "TRB", "AST", "PER", "WS", "BPM", "VORP"]
rows = []
for s in stats:
    r = {"Stat": STAT_LABELS[s],
         "Rookie": row[s], "Year 3": row[f"{s}_third"]}
    for tier in tiers:
        r[f"{TIER_SHORT[tier]} median (yr 3)"] = resolved.loc[resolved.Player_Label == tier, f"{s}_third"].median()
    rows.append(r)
table = pd.DataFrame(rows).set_index("Stat")
st.dataframe(table.style.format("{:.1f}"), width="stretch")

st.caption("Stats are season totals except PER, BPM and WS (season rate/total as published by "
           "Basketball-Reference). Players whose careers lasted fewer than three seasons aren't "
           "scored, because their third-season stats would overlap with the label.")
footer()

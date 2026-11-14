"""
Stage 3: build the modeling table (one row per player) from the warehouse.

    python -m nba_careers.etl table

Port of notebooks/data_extraction/Rookie_Stats_Getting_Data.ipynb. Feature and
label formulas are unchanged; the differences are that players are keyed by
id instead of name, stat types are selected by name, and two bookkeeping
columns are added:
  * bbref_id          stable player id
  * left_censored     True when the player's first season in the data is the
                      first season of the dataset (1976-77): they may be veterans,
                      not rookies, so they are excluded from training.
"""
from pathlib import Path

import duckdb
import polars as pl

from ..config import MODELING_TABLE
from .warehouse import WAREHOUSE_PATH

SEASON_STATS_SQL = """
SELECT p.player_id, p.bbref_id, p.name AS Player, s.label AS Season,
       st.name AS Stat_Name, v.value AS Stat_Value, t.stat_type_id
FROM player_stat_values v
JOIN players p    USING (player_id)
JOIN seasons s    USING (season_id)
JOIN stats st     USING (stat_id)
JOIN stat_types t USING (stat_type_id)
WHERE t.name IN ('player_totals', 'player_advanced')
"""


def _query(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    return pl.from_arrow(con.sql(sql).to_arrow_table())


def _min_max(col: str) -> pl.Expr:
    return (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())


def career_labels(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Career score and 3-tier label for every player (formula from the original notebook)."""
    per_season = _query(con, """
        WITH ps AS (
            SELECT v.player_id, v.season_id,
                   MAX(CASE WHEN st.name = 'WS'   THEN v.value END) AS Win_Share,
                   MAX(CASE WHEN st.name = 'USG%' THEN v.value END) AS Usage,
                   MAX(CASE WHEN st.name = 'G'    THEN v.value END) AS Games_Played,
                   MAX(CASE WHEN st.name = 'MP'   THEN v.value END) AS Minutes_Played,
                   MAX(CASE WHEN st.name = 'GS'   THEN v.value END) AS Games_Started,
                   MAX(CASE WHEN st.name = 'BPM'  THEN v.value END) AS Box_Plus_Minus
            FROM player_stat_values v
            JOIN stats st USING (stat_id)
            JOIN stat_types t USING (stat_type_id)
            WHERE t.name IN ('player_totals', 'player_advanced')
            GROUP BY v.player_id, v.season_id
        )
        SELECT ps.*, a.all_nba_team AS All_Nba_Team, a.allstar AS AllStar
        FROM award_info a JOIN ps USING (player_id, season_id)
    """)
    # Sorted + rounded so float sums don't depend on thread scheduling (byte-identical rebuilds)
    summary = per_season.sort(["player_id", "season_id"]).group_by("player_id", maintain_order=True).agg(
        pl.col("All_Nba_Team").max().alias("All_Nba_Team_Max"),
        pl.col("AllStar").max().alias("AllStar_Max"),
        pl.col("Games_Played").mean().alias("Games_Played_Mean"),
        pl.col("Win_Share").sum().alias("Win_Share_Sum"),
        pl.col("Box_Plus_Minus").mean().alias("Box_Plus_Minus_Mean"),
        pl.col("Minutes_Played").mean().alias("Minutes_Played_Mean"),
        pl.col("Games_Started").mean().alias("Games_Started_Mean"),
        pl.col("Usage").mean().alias("Usage_Mean"),
    )
    summary = summary.with_columns(
        (0.5 * _min_max("Win_Share_Sum") + 0.4 * _min_max("Box_Plus_Minus_Mean")
         + 0.4 * _min_max("Usage_Mean") + 0.2 * _min_max("Games_Started_Mean")
         + 0.1 * _min_max("Minutes_Played_Mean")).alias("Player_Score"))
    summary = summary.with_columns(pl.selectors.float().round(6))
    return summary.with_columns(
        pl.when(pl.col("Player_Score") > 0.4).then(pl.lit("Star Player"))
          .when(pl.col("Player_Score") > 0.3).then(pl.lit("Role Player"))
          .otherwise(pl.lit("Bust Player")).alias("Player_Label"))


def build(db_path: Path = WAREHOUSE_PATH, out_path: Path = MODELING_TABLE) -> pl.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    first_data_season = con.sql("SELECT MIN(label) FROM seasons s JOIN player_stat_values USING (season_id)").fetchall()[0][0]

    # ---- first three seasons, one row per (player, season), totals + advanced stats
    long = _query(con, SEASON_STATS_SQL)
    season_rank = (long.select("player_id", "Season").unique()
                       .with_columns(rank=pl.col("Season").rank("dense").over("player_id")))
    long = long.join(season_rank.filter(pl.col("rank") <= 3), on=["player_id", "Season"])
    first3 = (long.sort(["player_id", "Season", "stat_type_id", "Stat_Name"])
                  .pivot(on="Stat_Name", index=["player_id", "bbref_id", "Player", "Season"],
                         values="Stat_Value", aggregate_function="first")
                  .sort(["player_id", "Season"]))
    stat_cols = [c for c in first3.columns if c not in {"player_id", "bbref_id", "Player", "Season", "Player_Age"}]

    # ---- year-over-year growth (original formula: change relative to the later year)
    def growth(col: str, a: int, b: int) -> pl.Expr:
        x = pl.col(col)
        later, earlier = x.shift(-b).over("player_id"), (x.shift(-a).over("player_id") if a else x)
        return (later - earlier) / later * 100

    exprs = []
    for s in stat_cols:
        exprs += [growth(s, 0, 1).alias(f"{s}_growth_y1_y2"),
                  growth(s, 1, 2).alias(f"{s}_growth_y2_y3"),
                  ((growth(s, 0, 1) + growth(s, 1, 2)) / 2).alias(f"{s}_avg_yoy_growth")]
    rookie = (first3.fill_null(0).with_columns(exprs)
                    .filter(pl.col("Season") == pl.col("Season").min().over("player_id")))

    # ---- third (latest of the first three) season, suffixed _third
    third = (first3.filter(pl.col("Season") == pl.col("Season").max().over("player_id"))
                   .drop("bbref_id", "Player")
                   .rename({c: f"{c}_third" for c in first3.columns if c not in {"player_id", "bbref_id", "Player"}}))

    # ---- draft position and rookie award voting
    rookie_awards = _query(con, """
        SELECT p.player_id, p.draft_pick AS Draft_Pick, p.draft_class AS Draft_Class,
               r.rank AS ROY_Rank, r.first_place_votes AS ROY_Votes, r.point_share AS ROY_Point_Share,
               a.rank AS All_Rookie_Rank, a.first_place_votes AS All_Rookie_First_Place_Votes,
               a.second_place_votes AS All_Rookie_Second_Place_Votes, a.points_share AS All_Rookie_Point_Share
        FROM players p
        LEFT JOIN roy_voting r USING (player_id)
        LEFT JOIN all_rookie_voting a USING (player_id)
    """).unique("player_id", keep="first")

    # ---- rookie team and whether it made the playoffs the season before
    rookie_team = _query(con, """
        WITH rookie AS (
            SELECT player_id, MIN(s.label) AS label FROM player_stat_values JOIN seasons s USING (season_id)
            GROUP BY player_id
        ), team AS (
            SELECT DISTINCT v.player_id, v.team_id, s.label
            FROM player_stat_values v JOIN seasons s USING (season_id)
        )
        SELECT r.player_id, t.name AS Team, prev.made_playoffs AS team_made_playoffs_last_year
        FROM rookie r
        JOIN team tm ON tm.player_id = r.player_id AND tm.label = r.label
        JOIN teams t ON t.team_id = tm.team_id
        LEFT JOIN seasons sp ON sp.label = r.label - 1
        LEFT JOIN team_season_info prev ON prev.team_id = tm.team_id AND prev.season_id = sp.season_id
    """).unique("player_id", keep="first")

    labels = career_labels(con)
    con.close()

    table = (rookie.join(rookie_awards, on="player_id", how="left")
                   .join(rookie_team, on="player_id", how="left")
                   .join(third, on="player_id", how="left")
                   .join(labels, on="player_id", how="left")
                   .with_columns(left_censored=pl.col("Season") == first_data_season)
                   .drop("player_id")
                   .sort(["Season", "Player", "bbref_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_path)
    print(f"[table] {table.height:,} players x {table.width} columns -> {out_path}")
    return table

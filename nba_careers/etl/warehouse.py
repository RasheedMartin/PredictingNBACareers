"""
Stage 2: build the DuckDB warehouse from data/raw_data/.

    python -m nba_careers.etl warehouse

Replaces notebooks/data_extraction/ETL_Raw_To_Database.ipynb. Differences from
the notebook, all deliberate:
  * Players are keyed by Basketball-Reference id, not by cleaned name, so
    same-name players are no longer merged and punctuation variants
    ("J.J." vs "JJ") no longer split one player into two.
  * Every team code maps to a real franchise. Multi-team rows (2TM/3TM...) are
    "Multiple Teams" instead of all teams collapsing into one "Unknown".
  * Stat types have explicit names (player_totals, player_advanced, ...),
    so queries no longer depend on folder iteration order.
  * Combine names without a comma ("Almansa Izan") are parsed correctly.
  * The whole build is vectorized and deterministic (seconds, not minutes).
"""
import csv
import re
import unicodedata
from pathlib import Path

import duckdb
import polars as pl

from ..config import ROOT
from .fetch import RAW_DIR

WAREHOUSE_PATH = ROOT / "data/warehouse/nba_data.duckdb"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets", "BUF": "Buffalo Braves",
    "CHA": "Charlotte Bobcats", "CHH": "Charlotte Hornets", "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets", "DET": "Detroit Pistons", "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets", "IND": "Indiana Pacers", "KCK": "Kansas City Kings",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NJN": "New Jersey Nets", "NOH": "New Orleans Hornets", "NOJ": "New Orleans Jazz",
    "NOK": "New Orleans/Oklahoma City Hornets", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "NYN": "New York Nets", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "SDC": "San Diego Clippers", "SEA": "Seattle SuperSonics", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "VAN": "Vancouver Grizzlies", "WAS": "Washington Wizards",
    "WSB": "Washington Bullets",
    **{f"{n}TM": "Multiple Teams" for n in range(2, 6)}, "TOT": "Multiple Teams",
}

PLAYER_STAT_DIRS = {
    "player_totals": "NBA_Player_Data/Total_Player_Data",
    "player_per_game": "NBA_Player_Data/Per_Game_Data",
    "player_advanced": "NBA_Player_Data/Advanced_Data",
}
TEAM_STAT_DIRS = {
    "team_totals": "Team_Stats/Total_Team_Data",
    "team_per_game": "Team_Stats/Per_Game_Data",
    "team_advanced": "Team_Stats/Advanced_Data",
}
STAT_TYPES = list(PLAYER_STAT_DIRS) + list(TEAM_STAT_DIRS)
NON_STAT_COLS = {"Rk", "Player", "Team", "Tm", "Pos", "Awards", "Player-additional"}

COMBINE_RENAME = {
    "Hgt": "Height", "Wgt": "Weight", "Bmi": "Bmi", "Bf": "Body_Fat", "Wngspn": "Wingspan",
    "Stndrch": "Standing_Reach", "Handl": "Length_of_Hand", "Handw": "Width_of_Hand",
    "Stndvert": "Standing_Vertical", "Lpvert": "Max_Vertical_Leap", "Lane": "Lane_Agility_Time",
    "Shuttle": "Shuttle_Run", "Sprint": "3_Quarter_Sprint", "Bench": "Max_Bench_Press",
    "Bar": "Ration_of_Wingspan", "Pan": "Length_x_Width_Hand", "Pbhgt": "Standing_Reach_Vertical",
    "Pdhgt": "Standing_Reach_Max_Vertical",
}


# --------------------------------------------------------------------------- helpers
def clean_name(name: str | None) -> str | None:
    if name is None:
        return None
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", name.replace(".", "").replace("*", "")).strip()


def clean_name_expr(col: str = "Player") -> pl.Expr:
    return pl.col(col).map_elements(clean_name, return_dtype=pl.String)


def _first_year(path: Path) -> int:
    return int(re.search(r"(\d{4})", path.stem).group(1))


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _num(col: str) -> pl.Expr:
    return pl.col(col).str.strip_chars().str.replace_all(r"^\+", "").cast(pl.Float64, strict=False)


def _rank_int(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(r"\D", "").cast(pl.Int64, strict=False)


def _descriptions():
    """Stat descriptions written by the team (utils/dictionary_mapping.py), loaded by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dictionary_mapping", ROOT / "utils/dictionary_mapping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.player_combine_description, mod.player_stat_descriptions, mod.team_stat_descriptions


# --------------------------------------------------------------------------- readers
def read_player_stats(raw: Path = RAW_DIR) -> pl.DataFrame:
    """Long table: bbref_id, Player, season, team_abbr, Pos, stat_type, stat, value, Awards."""
    frames = []
    for stat_type, sub in PLAYER_STAT_DIRS.items():
        for path in sorted((raw / sub).glob("*.csv")):
            df = pl.read_csv(path, infer_schema_length=0)
            team_col = "Team" if "Team" in df.columns else "Tm"
            df = (df.rename({"Player-additional": "bbref_id", team_col: "team_abbr", "Age": "Player_Age"})
                    .filter(pl.col("bbref_id").is_not_null() & (pl.col("bbref_id") != ""))
                    .unique("bbref_id", keep="first", maintain_order=True)   # combined row for traded players
                    .with_columns(season=pl.lit(_first_year(path)), stat_type=pl.lit(stat_type)))
            if "Awards" not in df.columns:
                df = df.with_columns(Awards=pl.lit(None, pl.String))
            stat_cols = [c for c in df.columns if c not in NON_STAT_COLS | {"bbref_id", "team_abbr", "season", "stat_type"}]
            long = (df.select("bbref_id", "Player", "season", "team_abbr", "Pos", "Awards", "stat_type",
                              *[_num(c).alias(c) for c in stat_cols])
                      .unpivot(index=["bbref_id", "Player", "season", "team_abbr", "Pos", "Awards", "stat_type"],
                               variable_name="stat", value_name="value"))
            frames.append(long)
    return pl.concat(frames, how="vertical_relaxed").with_columns(clean_name_expr())


def read_team_stats(raw: Path = RAW_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(long stat table, team_season_info) keyed by full team name + season."""
    stat_frames, info_frames = [], []
    for stat_type, sub in TEAM_STAT_DIRS.items():
        for path in sorted((raw / sub).glob("*.csv")):
            rows = _read_rows(path)
            if stat_type == "team_advanced":
                # The group row ("Offense/Defense Four Factors") is misaligned in some
                # hand exports, so use the blank spacer columns instead:
                # [base stats] | offense four factors | defense four factors | arena info
                header, body = rows[1], rows[2:]
                groups, spacer_count = [], 0
                for h in header:
                    spacer_count += h == ""
                    groups.append(spacer_count)
                prefix = {1: "Team_", 2: "Opponent_"}
                header = [prefix.get(g, "") + h if h else "" for g, h in zip(groups, header)]
            else:
                header, body = rows[0], rows[1:]
            keep = [i for i, h in enumerate(header) if h]
            df = pl.DataFrame([[r[i] if i < len(r) else None for i in keep] for r in body],
                              schema=[header[i] for i in keep], orient="row")
            df = (df.filter(pl.col("Team") != "League Average")
                    .with_columns(made_playoffs=pl.col("Team").str.ends_with("*"),
                                  Team=pl.col("Team").str.replace_all("*", "", literal=True),
                                  season=pl.lit(_first_year(path))))
            info_cols = ["Arena", "Attend.", "Attend./G"]
            if set(info_cols) <= set(df.columns):
                info_frames.append(df.select(
                    "Team", "season", pl.col("Arena").alias("arena"),
                    _num("Attend.").cast(pl.Int64).alias("attendance"),
                    _num("Attend./G").cast(pl.Int64).alias("attendance_per_game"), "made_playoffs"))
            stat_cols = [c for c in df.columns if c not in {"Rk", "Team", "season", "made_playoffs", *info_cols}]
            stat_frames.append(df.select("Team", "season", pl.lit(stat_type).alias("stat_type"),
                                         *[_num(c).alias(c) for c in stat_cols])
                                 .unpivot(index=["Team", "season", "stat_type"], variable_name="stat",
                                          value_name="value"))
    return pl.concat(stat_frames, how="vertical_relaxed"), pl.concat(info_frames, how="vertical_relaxed")


def read_draft(raw: Path = RAW_DIR) -> pl.DataFrame:
    frames = []
    for path in sorted((raw / "Draft_Pick").glob("*.csv")):
        rows = _read_rows(path)
        header, body = rows[0], rows[1:]
        idx = {h: header.index(h) for h in ["Pk", "Tm", "Player", "College"]}
        id_idx = header.index("Player-additional") if "Player-additional" in header else None
        frames.append(pl.DataFrame({
            "draft_pick": [r[idx["Pk"]] for r in body],
            "draft_team": [TEAM_NAMES.get(r[idx["Tm"]], r[idx["Tm"]]) for r in body],
            "Player": [r[idx["Player"]] for r in body],
            "college": [r[idx["College"]] or None for r in body],
            "bbref_id": [(r[id_idx] or None) if id_idx is not None else None for r in body],
            "draft_class": [_first_year(path)] * len(body),
        }))
    return (pl.concat(frames, how="vertical_relaxed")
              .with_columns(pl.col("draft_pick").cast(pl.Int64, strict=False), pl.col("bbref_id").cast(pl.String),
                            clean_name_expr())
              .filter(pl.col("draft_pick").is_not_null()))


def read_bio(raw: Path = RAW_DIR) -> pl.DataFrame:
    """Birth year/place and college from the Kaggle players dataset (matched by name)."""
    path = raw / "NBA_Player_Data/Players.csv"
    if not path.exists():
        return pl.DataFrame(schema={"Player": pl.String, "birth_year": pl.Int64, "birth_city": pl.String,
                                    "birth_state": pl.String, "bio_college": pl.String})
    df = pl.read_csv(path, infer_schema_length=0)
    return (df.select(clean_name_expr("Player").alias("Player"),
                      pl.col("born").cast(pl.Int64, strict=False).alias("birth_year"),
                      pl.col("birth_city").alias("birth_city"), pl.col("birth_state").alias("birth_state"),
                      pl.col("collage").alias("bio_college"))
              .unique("Player", keep="first"))


def read_combine(raw: Path = RAW_DIR) -> pl.DataFrame:
    path = raw / "NBA_Draft_Combine/Draft Combine - Kaggle.csv"
    df = pl.read_csv(path, infer_schema_length=0, null_values=["NA", ""])
    df.columns = [c.title() for c in df.columns]

    def first_last(raw_name: str) -> str:
        raw_name = re.sub(r"\s+", " ", raw_name).strip()
        if "," in raw_name:
            last, first = [p.strip() for p in raw_name.split(",", 1)]
        else:  # "Almansa Izan" -> "Izan Almansa"
            parts = raw_name.split(" ")
            last, first = parts[0], " ".join(parts[1:])
        return clean_name(f"{first} {last}")

    df = df.rename({k: v for k, v in COMBINE_RENAME.items() if k in df.columns}).with_columns(
        pl.col("Player").map_elements(first_last, return_dtype=pl.String),
        pl.col("Year").cast(pl.Int64).alias("season"),
        pl.col("Ration_of_Wingspan").str.strip_suffix("%").cast(pl.Float64, strict=False) / 100,
    )
    stat_cols = [c for c in df.columns if c not in {"Player", "Year", "season", "Pos"}]
    return df.with_columns(*[pl.col(c).cast(pl.Float64, strict=False) for c in stat_cols])


def read_votes(raw: Path, sub: str, kind: str) -> pl.DataFrame:
    frames = []
    for path in sorted((raw / sub).rglob("*.csv")):
        df = pl.read_csv(path, infer_schema_length=0).rename({"-9999": "bbref_id"}, strict=False)
        frames.append(df.with_columns(season=pl.lit(_first_year(path)), kind=pl.lit(kind)))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(clean_name_expr()).filter(pl.col("bbref_id").is_not_null())


# --------------------------------------------------------------------------- build
def build(raw: Path = RAW_DIR, db_path: Path = WAREHOUSE_PATH) -> Path:
    stats = read_player_stats(raw)
    team_stats, team_info = read_team_stats(raw)
    draft, bio, combine = read_draft(raw), read_bio(raw), read_combine(raw)
    votes = {k: read_votes(raw, sub, k) for k, sub in [
        ("mvp", "MVP_Voting"), ("roy", "Rookie_OTY"), ("all_nba", "All_Nba_Data"), ("all_rookie", "All_Rookie_Team")]}

    # ---- seasons
    labels = sorted(set(stats["season"]) | set(team_stats["season"]) | set(combine["season"]))
    seasons = pl.DataFrame({"season_id": range(1, len(labels) + 1), "label": labels})
    season_id = dict(zip(seasons["label"], seasons["season_id"]))

    # ---- teams: every abbreviation seen; team-file names map back per season
    abbrs = set(stats["team_abbr"].drop_nulls()) | {a for v in votes.values() for a in v["Tm"].drop_nulls()}
    teams = pl.DataFrame({"abbreviation": sorted(abbrs)}).with_columns(
        name=pl.col("abbreviation").replace_strict(TEAM_NAMES, default=pl.col("abbreviation")))
    teams = teams.with_row_index("team_id", offset=1).with_columns(pl.col("team_id").cast(pl.Int64))
    team_id = dict(zip(teams["abbreviation"], teams["team_id"]))
    # (full name, season) -> abbreviation used by players that season
    name_season = (stats.select("team_abbr", "season").unique()
                        .with_columns(Team=pl.col("team_abbr").replace_strict(TEAM_NAMES, default=None))
                        .filter(pl.col("Team") != "Multiple Teams").drop_nulls()
                        .unique(["Team", "season"]))

    def attach_team(df: pl.DataFrame) -> pl.DataFrame:
        out = df.join(name_season, on=["Team", "season"], how="left")
        missing = out.filter(pl.col("team_abbr").is_null())["Team"].unique().to_list()
        if missing:
            print(f"[warehouse] warning: no team code for {missing[:5]}")
        return out.drop_nulls("team_abbr").with_columns(
            team_id=pl.col("team_abbr").replace_strict(team_id, return_dtype=pl.Int64))

    # ---- players (one per bbref id; first-seen name/season)
    first_seen = (stats.group_by("bbref_id").agg(pl.col("Player").first(), pl.col("season").min().alias("first_season"))
                       .sort(["first_season", "bbref_id"]))
    # draft: by id when the file has ids, otherwise by name near the draft year
    by_id = draft.filter(pl.col("bbref_id").is_not_null())
    by_name = (draft.filter(pl.col("bbref_id").is_null()).drop("bbref_id")
                    .join(first_seen, on="Player", how="inner")
                    .filter(pl.col("first_season").is_between(pl.col("draft_class"), pl.col("draft_class") + 3))
                    .drop("first_season"))
    draft_matched = pl.concat([by_id.drop("Player"), by_name.drop("Player").select(by_id.drop("Player").columns)]
                              ).unique("bbref_id", keep="first")
    # drafted players with ids but no NBA stats yet (e.g. this year's draft class)
    draft_only = by_id.join(first_seen, on="bbref_id", how="anti").select("bbref_id", "Player")
    players = (pl.concat([first_seen.select("bbref_id", "Player"), draft_only])
                 .join(draft_matched, on="bbref_id", how="left")
                 .join(bio, on="Player", how="left")
                 .with_columns(college=pl.coalesce("college", "bio_college")).drop("bio_college"))
    # combine prospects: match to an NBA player by name + draft year, else add a new player
    combine_names = combine.select("Player", "season").unique()
    cand = combine_names.join(players.select("bbref_id", "Player", "draft_class"), on="Player", how="left")
    cand = cand.join(first_seen.select("bbref_id", "first_season"), on="bbref_id", how="left").with_columns(
        ok=pl.col("bbref_id").is_not_null() & (
            (pl.col("draft_class") == pl.col("season")) |
            pl.col("first_season").is_between(pl.col("season"), pl.col("season") + 2) |
            pl.col("draft_class").is_null() & pl.col("first_season").is_null()))
    combine_ids = cand.filter("ok").unique(["Player", "season"]).select("Player", "season", "bbref_id")
    new_prospects = combine_names.join(combine_ids, on=["Player", "season"], how="anti").select("Player").unique()
    players = pl.concat([players, new_prospects.with_columns(bbref_id=pl.lit(None, pl.String))],
                        how="diagonal_relaxed")
    players = players.with_row_index("player_id", offset=1).with_columns(pl.col("player_id").cast(pl.Int64))
    pid_by_bbref = dict(zip(players["bbref_id"], players["player_id"]))
    prospect_pid = dict(zip(*players.filter(pl.col("bbref_id").is_null()).select("Player", "player_id")))

    # ---- stats & stat types
    player_combine_description, player_stat_descriptions, team_stat_descriptions = _descriptions()
    stat_names = sorted(set(stats["stat"]) | set(team_stats["stat"]) |
                        (set(combine.columns) - {"Player", "Year", "season", "Pos"}))
    desc = {**team_stat_descriptions, **player_combine_description, **player_stat_descriptions}
    stats_tbl = pl.DataFrame({"stat_id": range(1, len(stat_names) + 1), "name": stat_names,
                              "description": [desc.get(n, "") for n in stat_names]})
    stat_id = dict(zip(stats_tbl["name"], stats_tbl["stat_id"]))
    stat_types = pl.DataFrame({"stat_type_id": range(1, len(STAT_TYPES) + 1), "name": STAT_TYPES})
    stat_type_id = dict(zip(stat_types["name"], stat_types["stat_type_id"]))

    def ids(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            player_id=pl.col("bbref_id").replace_strict(pid_by_bbref, return_dtype=pl.Int64),
            season_id=pl.col("season").replace_strict(season_id, return_dtype=pl.Int64),
            team_id=pl.col("team_abbr").replace_strict(team_id, return_dtype=pl.Int64))

    psv = (ids(stats).with_columns(stat_type_id=pl.col("stat_type").replace_strict(stat_type_id, return_dtype=pl.Int64),
                                   stat_id=pl.col("stat").replace_strict(stat_id, return_dtype=pl.Int64))
                     .select("player_id", "team_id", "season_id", "stat_type_id", "stat_id", "value"))

    # ---- positions
    pos = (pl.concat([stats.select("bbref_id", "Pos").unique().drop_nulls()
                           .with_columns(player_id=pl.col("bbref_id").replace_strict(pid_by_bbref, return_dtype=pl.Int64))
                           .select("player_id", "Pos"),
                      combine.join(combine_ids, on=["Player", "season"], how="left")
                             .with_columns(player_id=pl.when(pl.col("bbref_id").is_not_null())
                                           .then(pl.col("bbref_id").replace_strict(pid_by_bbref, default=None, return_dtype=pl.Int64))
                                           .otherwise(pl.col("Player").replace_strict(prospect_pid, default=None, return_dtype=pl.Int64)))
                             .select("player_id", "Pos").drop_nulls()])
             .with_columns(pl.col("Pos").str.split("-")).explode("Pos")
             .filter(pl.col("Pos").is_in(["PG", "SG", "SF", "PF", "C", "G", "F"])))
    positions = pl.DataFrame({"name": sorted(set(pos["Pos"]))}).with_row_index("position_id", offset=1)
    player_positions = (pos.join(positions.rename({"name": "Pos"}), on="Pos")
                           .select("player_id", pl.col("position_id").cast(pl.Int64)).unique())

    # ---- awards parsed from the Awards column
    aw = (ids(stats.filter(pl.col("stat") == "G").unique(["bbref_id", "season"], keep="first"))
          .with_columns(a=pl.col("Awards").fill_null(""))
          .select("player_id", "season_id", "team_id",
                  *[pl.col("a").str.extract(rf"{k}-(\d+)", 1).cast(pl.Int64).fill_null(0).alias(v) for k, v in
                    [("MVP", "mvp_rank"), ("DPOY", "dpoy_rank"), ("ROY", "roy_rank"), ("6MOY", "six_moy_rank"),
                     ("CPOY", "comeback_player_rank"), ("MIP", "most_improved")]],
                  pl.col("a").str.contains("AS").cast(pl.Int64).alias("allstar"),
                  pl.col("a").str.extract(r"NBA(\d)", 1).cast(pl.Int64).fill_null(0).alias("all_nba_team"),
                  pl.col("a").str.extract(r"DEF(\d)", 1).cast(pl.Int64).fill_null(0).alias("all_defense_team")))

    # ---- team stats
    team_stats_ids = attach_team(team_stats).with_columns(
        season_id=pl.col("season").replace_strict(season_id, return_dtype=pl.Int64),
        stat_type_id=pl.col("stat_type").replace_strict(stat_type_id, return_dtype=pl.Int64),
        stat_id=pl.col("stat").replace_strict(stat_id, return_dtype=pl.Int64),
    ).select("team_id", "season_id", "stat_type_id", "stat_id", "value")
    tsi = attach_team(team_info).with_columns(
        season_id=pl.col("season").replace_strict(season_id, return_dtype=pl.Int64)
    ).select("team_id", "season_id", "arena", "attendance", "attendance_per_game", "made_playoffs")

    # ---- combine values
    combine_long = (combine.join(combine_ids, on=["Player", "season"], how="left")
                    .with_columns(player_id=pl.when(pl.col("bbref_id").is_not_null())
                                  .then(pl.col("bbref_id").replace_strict(pid_by_bbref, default=None, return_dtype=pl.Int64))
                                  .otherwise(pl.col("Player").replace_strict(prospect_pid, default=None, return_dtype=pl.Int64)),
                                  season_id=pl.col("season").replace_strict(season_id, return_dtype=pl.Int64))
                    .unpivot(index=["player_id", "season_id"],
                             on=[c for c in combine.columns if c not in {"Player", "Year", "season", "Pos", "bbref_id"}],
                             variable_name="stat", value_name="value")
                    .with_columns(stat_id=pl.col("stat").replace_strict(stat_id, return_dtype=pl.Int64))
                    .select("player_id", "season_id", "stat_id", "value")
                    .unique(["player_id", "season_id", "stat_id"], keep="first"))

    # ---- voting tables
    def vote_frame(kind: str, extra: list[pl.Expr], rank: pl.Expr) -> pl.DataFrame:
        v = votes[kind].filter(pl.col("bbref_id").is_in(list(pid_by_bbref)))
        return v.select(
            pl.col("bbref_id").replace_strict(pid_by_bbref, return_dtype=pl.Int64).alias("player_id"),
            pl.col("Tm").replace_strict(team_id, default=None, return_dtype=pl.Int64).alias("team_id"),
            pl.col("season").replace_strict(season_id, default=None, return_dtype=pl.Int64).alias("season_id"),
            rank.alias("rank"), *extra,
            _num("Pts Won").alias("points_won"), _num("Age").cast(pl.Int64).alias("age"),
            _num("Pts Max").alias("points_max"), _num("Share").alias(
                "points_share" if kind.startswith("all_") else "point_share"))

    tier = lambda col, n: (pl.when(pl.col(col).str.starts_with("1")).then(pl.lit("1ST Team"))  # noqa: E731
                           .when(pl.col(col).str.starts_with("2")).then(pl.lit("2ND Team"))
                           .when(pl.col(col).str.starts_with("3") & pl.lit(n == 3)).then(pl.lit("3RD Team"))
                           .when(pl.col(col) == "ORV").then(pl.lit("Received Votes"))
                           .otherwise(pl.lit("No Votes")))
    mvp = vote_frame("mvp", [_num("First").cast(pl.Int64).alias("first_place_votes")], _rank_int("Rank"))
    roy = vote_frame("roy", [_num("First").cast(pl.Int64).alias("first_place_votes")], _rank_int("Rank"))
    all_nba = vote_frame("all_nba", [_num("1st Tm").cast(pl.Int64).alias("first_place_votes"),
                                     _num("2nd Tm").cast(pl.Int64).alias("second_place_votes"),
                                     _num("3rd Tm").cast(pl.Int64).alias("third_place_votes")], tier("# Tm", 3))
    all_rookie = vote_frame("all_rookie", [_num("1st Tm").cast(pl.Int64).alias("first_place_votes"),
                                           _num("2nd Tm").cast(pl.Int64).alias("second_place_votes")], tier("# Tm", 2))

    # ---- write
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(SCHEMA_PATH.read_text())
    tables = {
        "players": players.select("player_id", "bbref_id", pl.col("Player").alias("name"), "birth_year", "college",
                                  "birth_state", "birth_city", "draft_pick", "draft_class", "draft_team"),
        "positions": positions.select(pl.col("position_id").cast(pl.Int64), "name"),
        "player_positions": player_positions,
        "teams": teams.select("team_id", "abbreviation", "name"),
        "seasons": seasons, "stat_types": stat_types, "stats": stats_tbl,
        "player_stat_values": psv, "award_info": aw,
        "team_stat_values": team_stats_ids.unique(["team_id", "season_id", "stat_type_id", "stat_id"]),
        "team_season_info": tsi.unique(["team_id", "season_id"]),
        "combine_stat_values": combine_long.drop_nulls("player_id"),
        "mvp_voting": mvp, "roy_voting": roy, "all_nba_voting": all_nba, "all_rookie_voting": all_rookie,
    }
    for name, df in tables.items():
        con.register("_df", df)
        cols = ", ".join(f'"{c}"' for c in df.columns)
        con.execute(f'INSERT INTO {name} ({cols}) SELECT {cols} FROM _df')
        con.unregister("_df")
    con.execute("CHECKPOINT")
    con.close()
    tmp.replace(db_path)
    print("[warehouse] " + ", ".join(f"{k}={v.height:,}" for k, v in tables.items()))
    return db_path


def export_app_db(src: Path = WAREHOUSE_PATH, dst: Path | None = None) -> Path:
    """The app only needs the combine tables; ship a small copy instead of the full warehouse."""
    from ..config import APP_DB
    dst = dst or APP_DB
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(f"ATTACH '{src}' AS w (READ_ONLY)")
    for t in ["combine_stat_values", "players", "player_positions", "positions", "seasons", "stats", "stat_types"]:
        con.execute(f"CREATE TABLE main.{t} AS SELECT * FROM w.main.{t}")
    con.execute("DETACH w")
    con.execute("CHECKPOINT")
    con.close()
    tmp.replace(dst)
    return dst

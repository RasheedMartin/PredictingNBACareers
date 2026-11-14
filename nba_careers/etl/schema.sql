-- NBA data warehouse (DuckDB). Rebuilt from scratch by `python -m nba_careers.etl warehouse`.
-- Table names match the original notebook schema so existing queries keep working.
-- Players are keyed by Basketball-Reference id (bbref_id) when one exists.

CREATE TABLE players (
    player_id    INTEGER PRIMARY KEY,
    bbref_id     VARCHAR UNIQUE,          -- NULL for combine-only prospects who never played
    name         VARCHAR NOT NULL,
    birth_year   INTEGER,
    college      VARCHAR,
    birth_state  VARCHAR,
    birth_city   VARCHAR,
    draft_pick   INTEGER,
    draft_class  INTEGER,
    draft_team   VARCHAR
);

CREATE TABLE positions (position_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);

CREATE TABLE player_positions (
    player_id   INTEGER,
    position_id INTEGER,
    PRIMARY KEY (player_id, position_id)
);

CREATE TABLE teams (
    team_id      INTEGER PRIMARY KEY,
    abbreviation VARCHAR NOT NULL UNIQUE, -- franchise-era code (SEA, OKC, ...) or 2TM/3TM for multi-team rows
    name         VARCHAR NOT NULL
);

CREATE TABLE seasons (
    season_id INTEGER PRIMARY KEY,
    label     INTEGER NOT NULL UNIQUE     -- season start year: 2024 = 2024-25
);

CREATE TABLE stat_types (stat_type_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);

CREATE TABLE stats (stat_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE, description VARCHAR);

CREATE TABLE player_stat_values (
    player_id    INTEGER NOT NULL,
    team_id      INTEGER NOT NULL,
    season_id    INTEGER NOT NULL,
    stat_type_id INTEGER NOT NULL,
    stat_id      INTEGER NOT NULL,
    value        DOUBLE,
    PRIMARY KEY (player_id, season_id, stat_type_id, stat_id)
);

CREATE TABLE award_info (
    player_id            INTEGER NOT NULL,
    season_id            INTEGER NOT NULL,
    team_id              INTEGER NOT NULL,
    mvp_rank             INTEGER,
    dpoy_rank            INTEGER,
    roy_rank             INTEGER,
    six_moy_rank         INTEGER,
    comeback_player_rank INTEGER,
    most_improved        INTEGER,
    allstar              INTEGER,
    all_nba_team         INTEGER,
    all_defense_team     INTEGER,
    PRIMARY KEY (player_id, season_id)
);

CREATE TABLE team_stat_values (
    team_id      INTEGER NOT NULL,
    season_id    INTEGER NOT NULL,
    stat_type_id INTEGER NOT NULL,
    stat_id      INTEGER NOT NULL,
    value        DOUBLE,
    PRIMARY KEY (team_id, season_id, stat_type_id, stat_id)
);

CREATE TABLE team_season_info (
    team_id             INTEGER NOT NULL,
    season_id           INTEGER NOT NULL,
    arena               VARCHAR,
    attendance          INTEGER,
    attendance_per_game INTEGER,
    made_playoffs       BOOLEAN,
    PRIMARY KEY (team_id, season_id)
);

CREATE TABLE combine_stat_values (
    player_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    stat_id   INTEGER NOT NULL,
    value     DOUBLE,
    PRIMARY KEY (player_id, season_id, stat_id)
);

CREATE TABLE mvp_voting (
    player_id INTEGER, team_id INTEGER, season_id INTEGER, rank INTEGER,
    first_place_votes INTEGER, points_won DOUBLE, age INTEGER, points_max DOUBLE, point_share DOUBLE
);
CREATE TABLE roy_voting (
    player_id INTEGER, team_id INTEGER, season_id INTEGER, rank INTEGER,
    first_place_votes INTEGER, points_won DOUBLE, age INTEGER, points_max DOUBLE, point_share DOUBLE
);
CREATE TABLE all_nba_voting (
    player_id INTEGER, team_id INTEGER, season_id INTEGER, rank VARCHAR,
    first_place_votes INTEGER, second_place_votes INTEGER, third_place_votes INTEGER,
    points_won DOUBLE, age INTEGER, points_max DOUBLE, points_share DOUBLE
);
CREATE TABLE all_rookie_voting (
    player_id INTEGER, team_id INTEGER, season_id INTEGER, rank VARCHAR,
    first_place_votes INTEGER, second_place_votes INTEGER,
    points_won DOUBLE, age INTEGER, points_max DOUBLE, points_share DOUBLE
);

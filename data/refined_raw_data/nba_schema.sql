CREATE TABLE IF NOT EXISTS players
(
    player_id INT AUTO_INCREMENT PRIMARY KEY,
    name
    VARHAR(150)
    NOT
    NULL,
    birth_year
    INTEGER,
    college
    VARCHAR(150),
    birth_state
    VARCHAR(150),
    birth_city
    TEXT,
    draft_pick INTEGER DEFAULT NULL,
    draft_class INTEGER DEFAULT NULL,
    draft_team TEXT DEFAULT NULL,
    UNIQUE
(
    name,
    birth_year,
    college
)

    );

CREATE TABLE IF NOT EXISTS positions
(
    position_id
    INTEGER
    PRIMARY
    KEY,
    name
    TEXT
    NOT
    NULL
    UNIQUE
);

CREATE TABLE IF NOT EXISTS player_positions
(
    player_id
    INTEGER,
    position_id
    INTEGER,

    PRIMARY
    KEY
(
    player_id,
    position_id
),
    FOREIGN KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    position_id
) REFERENCES positions
(
    position_id
)
    );

CREATE TABLE IF NOT EXISTS teams
(
    team_id
    INTEGER
    PRIMARY
    KEY,
    name
    TEXT,
    abbreviation
    TEXT
    NOT
    NULL
);

CREATE TABLE IF NOT EXISTS seasons
(
    season_id
    INTEGER
    PRIMARY
    KEY,
    label
    INTEGER
);

CREATE TABLE IF NOT EXISTS stat_types
(
    stat_type_id
    INTEGER
    PRIMARY
    KEY,
    name
    TEXT
    UNIQUE
);

CREATE TABLE IF NOT EXISTS stats
(
    stat_id
    INTEGER
    PRIMARY
    KEY,
    name
    TEXT
    UNIQUE,
    description
    TEXT
);

CREATE TABLE IF NOT EXISTS player_stat_values
(
    id
    INTEGER
    PRIMARY
    KEY,
    player_id
    INTEGER
    NOT
    NULL,
    team_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    stat_type_id
    INTEGER
    NOT
    NULL, -- e.g., "per_game", "total", "advanced"
    stat_id
    INTEGER
    NOT
    NULL, -- e.g., "points", "assists", etc.
    value
    REAL,

    FOREIGN
    KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    team_id
) REFERENCES teams
(
    team_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
),
    FOREIGN KEY
(
    stat_type_id
) REFERENCES stat_types
(
    stat_type_id
),
    FOREIGN KEY
(
    stat_id
) REFERENCES stats
(
    stat_id
)
    );

CREATE TABLE IF NOT EXISTS award_info
(
    player_id
    INTEGER,
    season_id
    INTEGER,
    team_id
    INTEGER,
    mvp_rank
    INTEGER,
    dpoy_rank
    INTEGER,
    roy_rank
    INTEGER,
    six_moy_rank
    INTEGER,
    most_improved
    INTEGER,
    comeback_player_rank
    INTEGER,
    allstar
    BOOLEAN,
    all_defense_team
    INTEGER,
    all_nba_team
    INTEGER,

    PRIMARY
    KEY
(
    player_id,
    season_id,
    team_id
),
    FOREIGN KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
),
    FOREIGN KEY
(
    team_id
) REFERENCES teams
(
    team_id
)
    );

CREATE TABLE IF NOT EXISTS team_stat_values
(
    id
    INTEGER
    PRIMARY
    KEY,
    team_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    stat_type_id
    INTEGER
    NOT
    NULL, -- e.g., "per_game", "total", "advanced"
    stat_id
    INTEGER
    NOT
    NULL,
    value
    REAL,

    FOREIGN
    KEY
(
    team_id
) REFERENCES teams
(
    team_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
),
FOREIGN KEY
(
    stat_type_id
) REFERENCES stat_types
(
    stat_type_id
),
    FOREIGN KEY
(
    stat_id
) REFERENCES stats
(
    stat_id
)
    );

CREATE TABLE IF NOT EXISTS team_season_info
(
    team_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    arena
    TEXT,
    attendance
    INTEGER,
    attendance_per_game
    REAL,
    made_playoffs
    BOOLEAN,

    PRIMARY
    KEY
(
    team_id,
    season_id
),
    FOREIGN KEY
(
    team_id
) REFERENCES teams
(
    team_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
)
    );


CREATE TABLE IF NOT EXISTS combine_stat_values
(
    id
    INTEGER
    PRIMARY
    KEY,
    player_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    stat_id
    INTEGER
    NOT
    NULL,
    value
    REAL,

    FOREIGN
    KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
),
    FOREIGN KEY
(
    stat_id
) REFERENCES stats
(
    stat_id
),
    UNIQUE
(
    player_id,
    season_id,
    stat_id
)
    );


CREATE TABLE IF NOT EXISTS mvp_voting
(
    vote_id
    INTEGER
    PRIMARY
    KEY,
    player_id
    INTEGER
    NOT
    NULL,
    team_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    rank
    INTEGER,
    first_place_votes
    INTEGER,
    points_won
    INTEGER,
    age
    INTEGER,
    points_max
    INTEGER,
    point_share
    REAL,

    FOREIGN
    KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    team_id
) REFERENCES teams
(
    team_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
)
    );

CREATE TABLE IF NOT EXISTS all_nba_voting
(
    vote_id
        INTEGER
        PRIMARY
            KEY,
    player_id
        INTEGER
        NOT
            NULL,
    team_id
        INTEGER
        NOT
            NULL,
    season_id
        INTEGER
        NOT
            NULL,
    rank
        TEXT,
    first_place_votes
        INTEGER,
    second_place_votes
        INTEGER,
    third_place_votes
        INTEGER,
    points_won
        INTEGER,
    age
        INTEGER,
    points_max
        INTEGER,
    points_share
        REAL,

    FOREIGN
        KEY
        (
         player_id
            ) REFERENCES players
        (
         player_id
            ),
    FOREIGN KEY
        (
         team_id
            ) REFERENCES teams
        (
         team_id
            ),
    FOREIGN KEY
        (
         season_id
            ) REFERENCES seasons
        (
         season_id
            )
);
CREATE TABLE IF NOT EXISTS all_rookie_voting
(
    vote_id
        INTEGER
        PRIMARY
            KEY,
    player_id
        INTEGER
        NOT
            NULL,
    team_id
        INTEGER
        NOT
            NULL,
    season_id
        INTEGER
        NOT
            NULL,
    rank
        TEXT,
    first_place_votes
        INTEGER,
    second_place_votes
        INTEGER,
    points_won
        INTEGER,
    age
        INTEGER,
    points_max
        INTEGER,
    points_share
        REAL,

    FOREIGN
        KEY
        (
         player_id
            ) REFERENCES players
        (
         player_id
            ),
    FOREIGN KEY
        (
         team_id
            ) REFERENCES teams
        (
         team_id
            ),
    FOREIGN KEY
        (
         season_id
            ) REFERENCES seasons
        (
         season_id
            )
);
CREATE TABLE IF NOT EXISTS roy_voting
(
    vote_id
    INTEGER
    PRIMARY
    KEY,
    player_id
    INTEGER
    NOT
    NULL,
    team_id
    INTEGER
    NOT
    NULL,
    season_id
    INTEGER
    NOT
    NULL,
    rank
    INTEGER,
    first_place_votes
    INTEGER,
    points_won
    INTEGER,
    age
    INTEGER,
    points_max
    INTEGER,
    point_share
    REAL,

    FOREIGN
    KEY
(
    player_id
) REFERENCES players
(
    player_id
),
    FOREIGN KEY
(
    team_id
) REFERENCES teams
(
    team_id
),
    FOREIGN KEY
(
    season_id
) REFERENCES seasons
(
    season_id
)
    );

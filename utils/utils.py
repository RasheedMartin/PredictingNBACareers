import io
from pathlib import Path

import pandas as pd
import polars as pl

nba_teams = {
    'GSW': 'Golden State Warriors',
    'SAC': 'Sacramento Kings',
    'MIN': 'Minnesota Timberwolves',
    'CLE': 'Cleveland Cavaliers',
    'WAS': 'Washington Wizards',
    'CHI': 'Chicago Bulls',
    'NYK': 'New York Knicks',
    'PHO': 'Phoenix Suns',
    'OKC': 'Oklahoma City Thunder',
    'NJN': 'New Jersey Nets',  # Former name of the Brooklyn Nets
    'LAC': 'Los Angeles Clippers',
    'PHI': 'Philadelphia 76ers',
    'SAS': 'San Antonio Spurs',
    'TOT': 'Multiple Teams',  # Used in stats for players traded mid-season
    'ATL': 'Atlanta Hawks',
    'DEN': 'Denver Nuggets',
    'ORL': 'Orlando Magic',
    'SEA': 'Seattle SuperSonics',  # Former team, became OKC Thunder
    'BRK': 'Brooklyn Nets',
    'TOR': 'Toronto Raptors',
    'DET': 'Detroit Pistons',
    'CHA': 'Charlotte Hornets',
    'CHO': 'Charlotte Hornets',
    'IND': 'Indiana Pacers',
    'MIA': 'Miami Heat',
    'BOS': 'Boston Celtics',
    'NOH': 'New Orleans Hornets',  # Now New Orleans Pelicans
    'UTA': 'Utah Jazz',
    'NOP': 'New Orleans Pelicans',
    'HOU': 'Houston Rockets',
    'MIL': 'Milwaukee Bucks',
    'POR': 'Portland Trail Blazers',
    'DAL': 'Dallas Mavericks',
    'MEM': 'Memphis Grizzlies',
    "VAN": "Vancouver Grizzlies",  # Now Memphis Grizzlies
    'LAL': 'Los Angeles Lakers',
}

player_stat_descriptions = {
    "3PA": "Three-Point Attempts per game or total",
    "G": "Games Played",
    "2P": "Two-Point Field Goals Made",
    "2PA": "Two-Point Field Goal Attempts",
    "PTS": "Points Scored",
    "eFG%": "Effective Field Goal Percentage (adjusts FG% to account for 3-pointers)",
    "FT": "Free Throws Made",
    "MP": "Minutes Played",
    "FGA": "Field Goal Attempts",
    "STL": "Steals",
    "3P%": "Three-Point Field Goal Percentage",
    "GS": "Games Started",
    "AST": "Assists",
    "TOV": "Turnovers",
    "2P%": "Two-Point Field Goal Percentage",
    "TRB": "Total Rebounds",
    "PF": "Personal Fouls",
    "3P": "Three-Point Field Goals Made",
    "DRB": "Defensive Rebounds",
    "FG%": "Field Goal Percentage",
    "FG": "Field Goals Made",
    "FTA": "Free Throw Attempts",
    "FT%": "Free Throw Percentage",
    "ORB": "Offensive Rebounds",
    "BLK": "Blocks",
    "Trp-Dbl": "Triple-Doubles (games with double digits in 3 stat categories)",
    "STL%": "Steal Percentage (estimate of opponent possessions ending in a steal by the player)",
    "TS%": "True Shooting Percentage (accounts for FG, 3P, and FT efficiency)",
    "WS/48": "Win Shares Per 48 Minutes (estimate of player contribution to wins per 48 minutes)",
    "PER": "Player Efficiency Rating (overall efficiency metric)",
    "BPM": "Box Plus/Minus (impact per 100 possessions relative to league average)",
    "3PAr": "Three-Point Attempt Rate (ratio of 3PA to FGA)",
    "TOV%": "Turnover Percentage (turnovers per 100 plays)",
    "DWS": "Defensive Win Shares",
    "BLK%": "Block Percentage (percentage of opponent shots blocked)",
    "DRB%": "Defensive Rebound Percentage",
    "OBPM": "Offensive Box Plus/Minus",
    "FTr": "Free Throw Rate (FTA per FGA)",
    "OWS": "Offensive Win Shares",
    "AST%": "Assist Percentage (percentage of teammate FGs assisted while on the floor)",
    "WS": "Win Shares (estimate of total wins contributed)",
    "ORB%": "Offensive Rebound Percentage",
    "USG%": "Usage Rate (estimate of team plays used by player while on floor)",
    "VORP": "Value Over Replacement Player",
    "DBPM": "Defensive Box Plus/Minus",
    "TRB%": "Total Rebound Percentage"
}

team_stat_descriptions = {
    "Rank": "Team rank within the league or conference",
    "Team": "Team name",
    "Age": "Average age of players on the team",
    "Win": "Number of wins during the season",
    "Lose": "Number of losses during the season",
    "PW": "Pythagorean wins (expected wins based on points scored and allowed)",
    "PL": "Pythagorean losses (expected losses based on points scored and allowed)",
    "MOV": "Margin of Victory (average point differential per game)",
    "SOS": "Strength of Schedule (average quality of opponents)",
    "SRS": "Simple Rating System (MOV adjusted for strength of schedule)",
    "ORtg": "Offensive Rating (points scored per 100 possessions)",
    "DRtg": "Defensive Rating (points allowed per 100 possessions)",
    "NRtg": "Net Rating (Offensive Rating minus Defensive Rating)",
    "Pace": "Average number of possessions per game",
    "FTr": "Free Throw Rate (FTA divided by FGA)",
    "3PAr": "Three-Point Attempt Rate (3PA divided by FGA)",
    "TS%": "True Shooting Percentage (accounts for FG, 3P, and FT efficiency)",
    "Team_eFG%": "Team Effective Field Goal Percentage",
    "Team_TOV%": "Team Turnover Percentage",
    "Team_ORB%": "Team Offensive Rebound Percentage",
    "Team_FT/FGA": "Team Free Throws Made per Field Goal Attempt",
    "Opponent_eFG%": "Opponent Effective Field Goal Percentage",
    "Opponent_TOV%": "Opponent Turnover Percentage",
    "Opponent_DRB%": "Opponent Defensive Rebound Percentage",
    "Opponent_FT/FGA": "Opponent Free Throws Made per Field Goal Attempt",
    "Arena": "Home arena of the team",
    "Total Attendance": "Total home game attendance for the season",
    "Avg_Home_Game_Attendance": "Average attendance per home game",
    "Season": "Season label (e.g., 2022-23)",
    "Playoffs": "Boolean indicating if the team made the playoffs"
}
player_combine_description = {
    "Year": "The year the data was recorded or the draft combine took place.",
    "Player": "Name of the player.",
    "Position": "Player's position on the basketball court (e.g., Guard, Forward, Center).",
    "Height": "Player's height in inches.",
    "Weight": "Player's weight in pounds.",
    "Bmi": "Body Mass Index calculated from height and weight.",
    "Body_Fat": "Body fat percentage of the player.",
    "Wingspan": "Player's wingspan (arm span) in inches.",
    "Standing_Reach": "The height in inches the player can reach while standing flat-footed.",
    "Length_of_Hand": "Length of the player's hand in inches.",
    "Width_of_Hand": "Width of the player's hand in inches.",
    "Standing_Vertical": "Vertical leap measurement from standing position in inches.",
    "Max_Vertical_Leap": "Maximum vertical leap in inches.",
    "Lane_Agility_Time": "Time in seconds for a lane agility drill (tests quickness and lateral movement).",
    "Shuttle_Run": "Time in seconds for shuttle run drill (tests lateral quickness and endurance).",
    "3_Quarter_Sprint": "Time in seconds for a 3/4 court sprint (tests speed and acceleration).",
    "Max_Bench_Press": "Maximum number of repetitions of 185 lbs bench press.",
    "Ration_of_Wingspan": "Ratio of wingspan to height.",
    "Length_x_Width_Hand": "Product of length and width of hand, representing hand size area.",
    "Standing_Reach_Vertical": "Sum of standing reach and standing vertical leap.",
    "Standing_Reach_Max_Vertical": "Sum of standing reach and maximum vertical leap."
}


def helper(table, drafted, weight, height, data_type, player) -> None:
    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
        df = pl.from_pandas(df)
    except Exception:
        return

    # Get name
    slug = player.split("/cbb/players/")[1].replace(".html", "")

    # Commenting out so we can identify nulls in Pos for aggregate vs by-season data
    # if "Pos" in df.columns:
    #     df = df.with_columns(
    #         pl.col("Pos").fill_null(strategy="forward")
    #     )
    df = df.with_columns(
        pl.lit(slug).alias("Name"),
        pl.lit("Drafted" if drafted else "Not drafted").alias("Drafted"),
        pl.lit(weight).alias("Weight"),
        pl.lit(height).alias("Height")
    )
    dir_path = Path(f"data/raw_data/College_Stats/{data_type}")
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / f"{slug}.csv"
    df.write_csv(filepath, separator=",")


def generate_full_polars_data_dictionary(df: pl.DataFrame, categorical_threshold: int = 20) -> list:
    total_rows = df.height
    result = []

    for col in df.columns:
        series = df[col]
        dtype = df.schema[col]
        non_null = series.drop_nulls()
        null_count = total_rows - non_null.len()
        percent_missing = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0
        unique_values = df.select(pl.col(col).n_unique())[0, 0]
        is_numeric = dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
        is_categorical = not is_numeric and unique_values <= categorical_threshold

        entry = {
            "column_name": col,
            "data_type": str(dtype),
            "non_null_count": non_null.len(),
            "null_count": null_count,
            "percent_missing": percent_missing,
            "unique_values": unique_values,
            "example_value": non_null[0] if non_null.len() > 0 else None,
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "description": ""  # You can manually fill this in later
        }

        # Numeric stats
        if is_numeric and non_null.len() > 0:
            stats = non_null.cast(pl.Float64).describe()
            entry.update({
                "min_value": non_null.min(),
                "max_value": non_null.max(),
                "mean": round(non_null.mean(), 4),
                "median": non_null.median(),
                "std_dev": round(non_null.std(), 4)
            })
        # Categorical mode
        elif is_categorical:
            mode = df.group_by(col).len().sort(col, descending=True)
            entry["mode"] = mode[0, col] if mode.height > 0 else None

        result.append(entry)

    return result
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
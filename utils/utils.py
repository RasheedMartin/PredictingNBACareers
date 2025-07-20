import io
from pathlib import Path

import pandas as pd
import polars as pl



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



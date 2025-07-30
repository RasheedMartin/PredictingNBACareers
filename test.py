from pathlib import Path
from utils.duckdb import get_duckdb_connection


print(Path("data/refined_raw_data/nba_data.duckdb").exists())
print(get_duckdb_connection())

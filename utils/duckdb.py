from pathlib import Path
import duckdb

def get_duckdb_connection(db_path=None, read_only=True) -> duckdb.DuckDBPyConnection:
    """
    Establishes a connection to a DuckDB database.
    Parameters:
        db_path (Path or str, optional): Path to the DuckDB database file. If None, defaults to
            '../../data/refined_raw_data/nba_data.duckdb' relative to the current working directory.
        read_only (bool, optional): If True, opens the database in read-only mode. Defaults to True.
    Returns:
        duckdb.DuckDBPyConnection: A connection object to the specified DuckDB database.
    Raises:
        FileNotFoundError: If the specified DuckDB database file does not exist.
    Prints:
        The resolved path to the DuckDB database upon successful connection.
    """
    if db_path is None:
        db_path = Path.cwd().parent.parent / "data/refined_raw_data/nba_data.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {db_path}")
    print(f"Connected to: {db_path.resolve()}")
    return duckdb.connect(db_path, read_only=read_only)
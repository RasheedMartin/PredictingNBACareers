from pathlib import Path
import duckdb
import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    probplot, 
    ttest_ind, 
    mannwhitneyu, 
    pearsonr, 
    f_oneway
)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to sys.path

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


@st.cache_data(ttl=3600)
def load_data(filters=None):
    with get_duckdb_connection(Path("data/refined_raw_data/nba_data.duckdb")) as conn:
        
        query = """
            SELECT 
            p.name as Player,
            po.name as Position,
            s.label as Season,
            st.name as Stat,
            csv.value as Value

            FROM nba_data.main.combine_stat_values csv
            JOIN players p ON csv.player_id = p.player_id
            FULL OUTER JOIN player_positions pp ON pp.player_id = p.player_id
            FULL OUTER JOIN positions po on po.position_id = pp.position_id
            JOIN seasons s ON csv.season_id = s.season_id
            JOIN stats st on csv.stat_id = st.stat_id
            """
        
        if filters:
            conditions = []
            params = []
            

            positions = filters.get("positions", [])
            if positions:
                conditions.append(f"po.name IN ({', '.join(['?'] * len(positions))})")
                params.extend(positions)
            seasons = filters.get("year_range", None)
            if seasons:
                start_year, end_year = seasons
                valid_years = [f"{year}" for year in range(start_year, end_year + 1)]
                seasons_placeholder = ', '.join(['?'] * len(valid_years))
                conditions.append(f"s.label IN ({seasons_placeholder})")
                params.extend(valid_years)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
    
            query += " ORDER BY p.name, s.label"
            print(query)
            return pl.from_arrow(conn.execute(query, params).arrow())
        return pl.from_arrow(conn.execute(query).arrow())
    

def univariate_eda(df: pl.DataFrame, column=None, bins=30):
    """
    Perform univariate analysis on specified numerical columns.
    If columns is None, all numeric columns will be analyzed.
    """

    st.write(f"\n--- {column} ---")
    st.write(df[column].describe())
    st.write(f"Skewness: {df[column].skew():.2f}")
    st.write(f"Kurtosis: {df[column].kurtosis():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Histogram + KDE
    sns.histplot(df[column], bins=bins, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"{column} Distribution")

    # Boxplot
    sns.boxplot(x=df[column], ax=axes[1], color='salmon')
    axes[1].set_title(f"{column} Boxplot")

    # QQ Plot for normality check

    probplot(df[column].drop_nulls(), dist="norm", plot=axes[2])
    axes[2].set_title(f"{column} QQ Plot")

    plt.tight_layout()
    return fig


# Function to perform statistical tests
def perform_statistical_tests(df: pl.DataFrame, col1: str, col2: str):
    """
    Perform t-test, Mann-Whitney U test, and Pearson correlation between two columns.
    """
    # Drop rows where either col1 or col2 is null
    df_clean = df.filter(
        pl.col(col1).is_not_null() & pl.col(col2).is_not_null()
    )

    # Extract values as NumPy arrays
    col1_data = df_clean[col1].to_numpy()
    col2_data = df_clean[col2].to_numpy()

    # T-test
    t_stat, p_value_ttest = ttest_ind(col1_data, col2_data, equal_var=False)

    # Mann-Whitney U test
    u_stat, p_value_mannwhitney = mannwhitneyu(col1_data, col2_data, alternative="two-sided")

    # Pearson correlation (optional)
    pearson_corr, _ = pearsonr(col1_data, col2_data)

    return {
        "t_stat": t_stat,
        "p_value_ttest": p_value_ttest,
        "u_stat": u_stat,
        "p_value_mannwhitney": p_value_mannwhitney,
        "pearson_corr": pearson_corr
    }

def plot_stats_by_position(col: str, df: pl.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    # Create a bar plot with error bars
    # Using seaborn's barplot to get mean and standard deviation
    # Note: Seaborn's barplot automatically calculates the mean and standard deviation
    # but we will add error bars manually for better control
    ax = sns.barplot(
        df,
        x="Position",
        y=f"{col}_mean",
        capsize=5,
        alpha=0.7,
        palette="pastel",
        hue="Position",
    )
    
    # Add error bars manually using matplotlib
    x_coords = range(len(df))
    means = df[f"{col}_mean"].to_list()
    stds = df[f"{col}_std"].to_list()
    
    ax.errorbar(x_coords, means, yerr=stds, fmt='none', c='black', capsize=5, lw=1.5)

      # Add std text on top of each bar
    for x, y, std in zip(x_coords, means, stds):
        offset = y * 0.01 # Slightly above the error bar
        ax.text(
            x,
            y + std + offset,
            f"±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            fontweight="bold"
        )

    ax.set_ylabel(f"{col.replace('_', ' ')} (Mean ± Std)")
    ax.set_title(f"{col.replace('_', ' ')} by Position")
    ax.set_xticks(x_coords)

    plt.ylabel(f"{col} (Mean ± Std)")
    plt.title(f"{col} by Position")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, df.select("Position", f"{col}_mean", f"{col}_std")


def combine_stat_tests_by_position(df: pl.DataFrame, stat=None, position_col="Position"):
    """
    Perform ANOVA and pairwise t-tests for each combine stat across positions.
    Returns a summary DataFrame with p-values for ANOVA and t-tests.
    """
    if stat is None:
        return
    positions = df[position_col].unique().to_list()
    results = []

    # Prepare data for ANOVA
    groups = [df.filter(pl.col(position_col) == pos)[stat].drop_nulls().to_numpy() for pos in positions]
    # Only run ANOVA if there are at least two groups with data
    if sum(len(g) > 1 for g in groups) >= 2:
        anova_p = f_oneway(*groups).pvalue
    else:
        anova_p = None

    # Pairwise t-tests
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i >= j:
                continue
            g1 = groups[i]
            g2 = groups[j]
            if len(g1) > 1 and len(g2) > 1:
                ttest_p = ttest_ind(g1, g2, nan_policy='omit').pvalue
            else:
                ttest_p = None
            results.append({
                "Stat": stat,
                "Position1": pos1,
                "Position2": pos2,
                "ANOVA_p": anova_p,
                "TTest_p": ttest_p
            })
    return pl.from_dicts(results)


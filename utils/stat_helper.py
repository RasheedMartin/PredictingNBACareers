import polars as pl
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr

# Functions to be shared across the project

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

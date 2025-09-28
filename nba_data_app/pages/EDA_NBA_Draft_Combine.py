from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import polars as pl
import seaborn as sns
from utils._utils import (
    load_data, 
    get_duckdb_connection,
    univariate_eda,
    perform_statistical_tests,
    plot_stats_by_position,
    combine_stat_tests_by_position
)

### App Layout
st.set_page_config(
    page_title="NBA Draft Combine EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Exploratory Data Analysis: NBA Draft Combine")


page = st.sidebar.selectbox("Select a Page", ["Overview", "Seasons Distribution", "Univariate Analysis", "Bivariate Analysis"])
st.sidebar.markdown("---")
st.sidebar.markdown("[📂 View this project on GitHub](https://github.com/RasheedMartin/PredictingNBACareers)")

# Sidebar page selector


if page == "Overview":
    st.markdown("""
---
**Author:** Rasheed Martin

This notebook provides a comprehensive exploratory analysis of the NBA Draft Combine dataset, focusing on uncovering key relationships between physical measurements, combine performance, and NBA career outcomes. The workflow includes:

- **Data Extraction & Cleaning:** Loading combine stats, NBA performance metrics, and award information for thousands of players and seasons.
- **Univariate & Bivariate Analysis:** Summarizing distributions, missingness, and correlations among physical and performance features.
- **Visualization:** Using barplots, heatmaps, pairplots, and boxplots to reveal trends by position, season, and award status.
- **Key Insights:** Identifying which combine metrics are most predictive of NBA success, and how standout physical traits relate to accolades such as All-Star, All-NBA, and Defensive Team selections.

This analysis supports downstream modeling and provides actionable insights for scouting, player development, and understanding the impact of draft combine results on NBA careers."""
)

    st.sidebar.title('Filters')

    year_range = st.sidebar.slider('Year Range', 2000, 2025, (2005, 2020))
    positions = st.sidebar.multiselect('Positions', ['PG', 'SG', 'PF', 'SF', 'C'], default=['PG', 'SF', 'C'])

    filters = {
        'year_range': year_range,
        'positions': positions,
    }

    dataframe = st.empty()
    # Lets turn the data from long to wide format
    raw_data = load_data(filters)
    data = raw_data.group_by(
        ['Player', 'Season', 'Value', 'Stat'],  # all other columns
    ).agg(pl.col("Position").unique().sort().alias("Position"))

    data = data.with_columns(
        pl.col("Position").list.join("/")
    )
    data: pl.DataFrame = data.pivot(
        values="Value",
        index=["Position", "Player", "Season"],
        on="Stat"
    )

    dataframe.dataframe(data.sort("Season", descending=True))
    st.markdown("---")
    st.subheader("Data Dictionary")
    with get_duckdb_connection(Path("data/refined_raw_data/nba_data.duckdb")) as conn:
        stat_columns = data.drop("Position", "Season", "Player").columns

        data_dictionary = pl.from_arrow(conn.execute(
            """
            SELECT s.name as Stat, s.description as Description
            FROM nba_data.main.combine_stat_values csv
            JOIN nba_data.main.stats s ON s.stat_id = csv.stat_id
            """
        ).arrow()).unique("Stat").sort("Stat")
        st.dataframe(data_dictionary)

        st.subheader("Stat Descriptions")
        st.write(data.select(stat_columns).describe())

elif page == "Seasons Distribution":
    st.markdown("---")
    st.subheader("Distribution of Players Across Seasons")
    tab1, tab2 = st.tabs(['CountPlot', 'Draft Combine Participation'])

    # Lets turn the data from long to wide format
    raw_data = load_data()
    data = raw_data.group_by(
        ['Player', 'Season', 'Value', 'Stat'],  # all other columns
    ).agg(pl.col("Position").unique().sort().alias("Position"))

    data = data.with_columns(
        pl.col("Position").list.join("/")
    )
    data: pl.DataFrame = data.pivot(
        values="Value",
        index=["Position", "Player", "Season"],
        on="Stat"
    )



    with tab1:
        st.header('Number of Player Records by Season')
        seasons_plot = st.empty()
        # Plotting the distribution of players across seasons
        fig = plt.figure(figsize=(14, 6))
        ax = sns.countplot(data["Season"], color="skyblue", order=sorted(data["Season"].unique()))
        # Add value labels on top of bars
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width)}",  # label
                        (width + 1, p.get_y() + p.get_height() / 2),  # position
                        ha='left', va='center', fontsize=9)

        # Showing the distribution across seasons
        plt.ylabel("Number of Players", fontsize=12)
        plt.xlabel("Season", fontsize=12)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_xticklabels([])
        plt.grid(axis="x")
        plt.tight_layout()
        seasons_plot.pyplot(fig)

        st.markdown("""
            The distribution of players across seasons shows how many records were collected for each season in the NBA.
            As observed above, the number of NBA Draft Combine records per season fluctuates rather than following a linear trend. 
            Prior to 2009, participation consistently exceeded 70 players per season, but since then, attendance has varied considerably. 
            This could reflect changes in data collection, combine invitation policies, or shifts in player interest.

            To better understand these patterns, let's visualize the trend in combine participation over time using a line graph. 
            This will help us identify periods of high and low attendance and assess the overall trajectory of combine involvement across seasons.
        """)

    with tab2:
        st.header('Trend in Draft Combine Participation by Season')
        # Histogram, boxplots

                # 1. Get number of total records
        total_seasons = len(data)

        # 2. Compute relative frequency or counts by season (As a record counts as attendance)
        avg_attendance = (
            data
            .group_by("Season")
            .agg((pl.len() / total_seasons).alias("Proportion"))
            .sort("Season")
        )

        # 3. Convert season to numeric for trendline
        seasons_str = avg_attendance["Season"].to_list()
        try:
            seasons = [int(s[:4]) for s in seasons_str]  # Because format "2003-04"
        except:
            seasons = list(range(len(seasons_str)))      # fallback

        values = avg_attendance["Proportion"].to_list()

        # 4. Fit linear regression
        coeffs = np.polyfit(seasons, values, 1)
        trend = np.poly1d(coeffs)

        # 5. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(seasons, values, marker='o', label="Proportion")
        plt.plot(seasons, trend(seasons), linestyle="--", color="red", label="Trendline")

        plt.xlabel("Season")
        plt.ylabel("Proportion of Records")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown(
        """
            #### Insights from the Season Distribution Plot

            - **Combine Participation is Limited:** The NBA Draft Combine consistently attracts fewer than 100 players per season, with participation rates fluctuating rather than following a clear upward or downward trend.
            - **No Linear Trend:** There is no steady increase or decrease in attendance over the years. Instead, some seasons see spikes or drops, likely due to changes in invitation policies, draft class size, or data collection practices.
            - **Recent Variability:** In more recent years, attendance appears more variable, suggesting that the combine may be less standardized or that external factors (e.g., league changes, pandemic disruptions) have influenced participation.
            - **Implications for Analysis:** The limited and inconsistent coverage means that combine data may not represent all NBA prospects equally. Insights drawn from this dataset should be interpreted with caution, as trends may not generalize to the entire player pool.

            Overall, while the combine provides valuable physical and athletic measurements, its selective nature and fluctuating participation highlight the importance of considering sample bias in downstream analysis.

            """
        )

elif page == "Univariate Analysis":
    st.markdown("---")
    st.subheader("Univariate Analysis of Combine Stats")
    st.markdown(
        """
        Univariate analysis examines each feature in the NBA Draft Combine dataset individually to understand its distribution, central tendency, spread, and potential outliers. 
        This step is crucial for identifying data quality issues, skewness, and the presence of missing values before proceeding to more complex analyses.

        **Key steps in this section:**
        - Visualize the distribution of each numerical feature using histograms, boxplots, and QQ plots.
        - Calculate summary statistics such as mean, median, standard deviation, skewness, and kurtosis.
        - Identify features with high missingness or unusual distributions that may require special handling.
        - Highlight outliers and data entry errors (e.g., minimum values of zero in physical measurements).

        This foundational analysis provides insights into the typical physical and athletic profiles of NBA prospects, 
        reveals which features are well-behaved, and flags variables that may need cleaning or imputation. 
        The results guide further data preparation and inform downstream modeling and bivariate analysis.
        """
    )
    st.markdown("---")

    
    # Load data
    raw_data = load_data()
    data = raw_data.group_by(
        ['Player', 'Season', 'Value', 'Stat'],  # all other columns
    ).agg(pl.col("Position").unique().sort().alias("Position"))

    data = data.with_columns(
        pl.col("Position").list.join("/")
    )
    data: pl.DataFrame = data.pivot(
        values="Value",
        index=["Position", "Player", "Season"],
        on="Stat"
    )
    df_wide_view_positions = data.with_columns(
        pl.when(pl.col("Position").str.contains("/", literal=True))
        .then(pl.col("Position").str.split("/"))
        .otherwise(pl.col("Position").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)))
        .replace("", None)
        .alias("Position")
    ).explode("Position")

    df_wide_view_positions = df_wide_view_positions.drop_nulls("Position")
    with st.expander("Data Overview"):
        tab, = st.tabs(['Feature Distribution'])

        with tab:
            st.subheader("Feature Distribution")
            st.markdown(
                """
                This section provides an overview of the distribution of key physical and performance metrics from the NBA Draft Combine dataset. 
                We will analyze features such as height, weight, vertical leap, and agility scores to understand their central tendencies, spreads, and potential outliers.
                """
            )
            st.markdown("---")
            st.subheader("Select a Feature for Univariate Analysis")

            

            column = st.selectbox("Select a Column", data.drop(["Position", "Player", "Season"]).columns)
            if column:
                st.markdown(f"### Univariate Analysis for {column}")
                plot = st.empty()
                fig = univariate_eda(data, column=column)
                plot.pyplot(fig)
elif page == "Bivariate Analysis":
    st.markdown("---")
    st.subheader("Bivariate Analysis of Combine Stats")
    st.markdown(
        """
        This section explores relationships between numerical features and the categorical variable in the dataset: **position**.
        - **Numerical vs. Numerical:**  
            We analyze correlations among all combine metrics to identify which physical and athletic measurements are most strongly related. This includes visualizing the correlation matrix and highlighting the strongest feature pairs.

        - **Numerical vs. Categorical:**  
            We examine how each combine statistic varies by player position (PG, SG, SF, PF, C). This involves:
            - Comparing means and distributions of key metrics across positions using barplots, boxplots, and statistical tests (ANOVA, t-tests).
            - Visualizing trends in combine measurements over time to uncover shifts in player profiles or combine participation.

        These analyses help reveal which physical and performance traits distinguish player roles, how combine metrics have evolved, and which features are most predictive of NBA success.
        """
    )
    st.markdown("---")

    with st.expander("Data Overview"):
        tab = st.tabs(['Top 15 Correlations', 'Comparison by Position'])
        tab1, tab2 = tab
        

        
        # Load data
        raw_data = load_data()
        data = raw_data.group_by(
            ['Player', 'Season', 'Value', 'Stat'],  # all other columns
        ).agg(pl.col("Position").unique().sort().alias("Position"))

        data = data.with_columns(
            pl.col("Position").list.join("/")
        )
        data: pl.DataFrame = data.pivot(
            values="Value",
            index=["Position", "Player", "Season"],
            on="Stat"
        )
        df_wide_view_positions = data.with_columns(
            pl.when(pl.col("Position").str.contains("/", literal=True))
            .then(pl.col("Position").str.split("/"))
            .otherwise(pl.col("Position").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)))
            .replace("", None)
            .alias("Position")
        ).explode("Position")

        df_wide_view_positions = df_wide_view_positions.drop_nulls("Position")

        with tab1:
            corr_matrix = data.drop(["Position", "Season", "Player"]).drop_nulls().corr()

            with st.expander("Correlation Matrix"):
                st.markdown(
                    """
                    The correlation matrix provides insights into the relationships between different physical and performance metrics from the NBA Draft Combine dataset. 
                    By examining how features correlate with each other, we can identify potential multicollinearity issues and understand which metrics are most predictive of NBA success.
                    """
                )
                st.markdown("---")

                

                # Plot heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
                            cmap="coolwarm", linewidths=0.5,
                            xticklabels=corr_matrix.columns,
                            yticklabels=corr_matrix.columns
                            )

                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)
            st.subheader("Top 15 Correlations")
            st.markdown(
                """
                This section highlights the top 15 strongest correlations between different combine metrics. 
                Understanding these relationships helps identify which features are most closely related and may influence each other.
                """
            )
            # Calculate correlation matrix
            corr_long = (
                corr_matrix.unpivot(
                    variable_name="Feature1",
                    value_name="Correlation"
                )
                .with_columns([
                    pl.repeat(corr_matrix.columns, corr_matrix.height).explode().alias("Feature2")
                ])
            )
            corr_filtered = (
                corr_long
                .filter(pl.col("Feature1") != pl.col("Feature2")) # remove self pairs
                .filter(pl.col("Feature1") < pl.col("Feature2")) # locks in on upper triangle
            )

            corr_final = corr_filtered.with_columns(
                pl.col("Correlation").abs().alias("AbsCorrelation")
            )
            top_corrs = corr_final.sort("AbsCorrelation", descending=True).head(15)

            top_corrs = top_corrs.with_columns(
                pl.col("Feature1").str.replace_all("_", " ").str.to_titlecase().alias("Feature1"),
                pl.col("Feature2").str.replace_all("_", " ").str.to_titlecase().alias("Feature2")
            )
            st.dataframe(top_corrs)
            st.markdown(
                """
                The correlation matrix reveals the strength and direction of relationships between different combine metrics. 
                High absolute correlation values (close to 1 or -1) indicate strong linear relationships, while values near 0 suggest weak or no linear relationship.
                
                The top 15 strongest correlations are displayed above, highlighting pairs of features that are most closely related. 
                This information can guide feature selection and engineering in predictive modeling tasks.
                """
            )
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=top_corrs,
                x="AbsCorrelation",
                y=top_corrs["Feature1"] + " vs " + top_corrs["Feature2"],
                palette="pastel",
                hue=top_corrs["Feature1"],
                legend=False
            )
            plt.title(f"Top {15} Strongest Correlations Between Combine Metrics")
            plt.xlabel("Absolute Correlation")
            plt.ylabel("Feature Pairs")
            plt.tight_layout()
            st.pyplot(plt)
            st.markdown(
                """
                    This is insightful! It's fascinating to see how these features interrelate—attributes like height, standing reach, wingspan, and weight are all closely connected. 
                    The taller the player, the higher the standing reach and typically the greater the wingspan and weight. 
                    These physical traits naturally influence performance metrics such as rebounding and blocking, as confirmed by the strong correlations observed. 
                    Understanding these relationships helps explain why certain combine measurements are so predictive of NBA success. 
                    Let's continue to explore how these attributes impact player outcomes and career achievements.
                """
            )

        with tab2:

            numeric_cols = df_wide_view_positions.select([pl.col(pl.Float32), pl.col(pl.Float64), pl.col(pl.Int64)]).columns

            df_summary = df_wide_view_positions.group_by("Position").agg(
                [
                    *(pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols),
                    *(pl.col(col).std().alias(f"{col}_std") for col in numeric_cols)
                ]
            )

            st.subheader("Comparison by Position")
            st.markdown(
                """
                This section compares combine metrics across different player positions to identify how physical attributes vary by role. 
                We will visualize the distributions of key metrics for each position and highlight any significant differences.
                """
            )
            st.markdown("---")
            stat_columns = data.drop(["Position", "Player", "Season"]).columns

            # Initialize session state if not already set
            if "stat" not in st.session_state:
                st.session_state.stat = stat_columns[0]

            # Create selectboxes
            st.selectbox("Select Features", stat_columns, index=stat_columns.index(st.session_state.stat), key="stat")

            # Update session state manually
            stat = st.session_state.stat

            if stat:
                st.markdown(f"### Univariate Analysis for {stat}")
                fig, df_stats = plot_stats_by_position(stat, df_summary)
                st.pyplot(fig)
                st.markdown(
                    """
                    The plot above shows the distribution of the selected combine metric across different player positions. 
                    This visualization helps identify how physical attributes vary by role, revealing trends and differences in athletic profiles among positions.
                    """
                )
                with st.expander("View Summary Statistics"):
                    st.dataframe(df_stats)

    with st.expander("Statistical Tests"):
            tab = st.tabs(['Numerical vs Numerical', 'Categorical vs Numerical'])
            tab1, tab2 = tab
            with tab1:
                st.subheader("Statistical Tests: Numerical vs Numerical")
                st.markdown(
                    """
                    This section performs statistical tests to compare the distributions of different combine metrics. 
                    We will conduct t-tests, Mann-Whitney U tests, and Pearson correlation analyses to assess the significance of differences between features.
                    """
                )
                st.markdown("---")

                columns = data.drop(["Position", "Player", "Season"]).columns

                # Initialize session state if not already set
                if "col1" not in st.session_state:
                    st.session_state.col1 = columns[0]

                if "col2" not in st.session_state:
                    st.session_state.col2 = columns[1] if len(columns) > 1 else columns[0]

                # Create selectboxes
                st.selectbox("Select First Column", columns, index=columns.index(st.session_state.col1), key="col1")
                st.selectbox("Select Second Column", columns, index=columns.index(st.session_state.col2), key="col2")

                # Update session state manually
                col1 = st.session_state.col1
                col2 = st.session_state.col2

                if col1 and col2:
                    st.markdown(f"### Statistical Tests between {col1} and {col2}")
                    results = perform_statistical_tests(data, col1, col2)
                    st.json(results)

                st.markdown("---")
                st.markdown("""
                    ### Insights from Statistical Tests on NBA Draft Combine Metrics

                    - **Strong Feature Relationships:** Many combine metrics show statistically significant relationships, confirmed by extremely low p-values in both t-tests and Mann-Whitney tests. 
                            For example, Height and Weight (p ≈ 0), Standing Reach and Wingspan (p ≈ 0), and Length_x_Width_Hand and Width_of_Hand (p ≈ 0) are highly correlated, 
                            reflecting underlying physical connections.

                    - **Physical Traits Cluster Together:** Features related to size—Height, Weight, Wingspan, Standing Reach—consistently exhibit strong positive correlations (Pearson r > 0.7). 
                            This means taller players tend to have longer wingspans, greater reach, and higher body mass.

                    - **Performance Metrics Interlinked:** Athletic performance stats such as Max Vertical Leap and Standing Vertical are tightly coupled (r ≈ 0.85), 
                            indicating that explosive leaping ability is captured by multiple combine measures.

                    - **Hand Measurements:** Length_x_Width_Hand, Width_of_Hand, and Length_of_Hand are all strongly correlated (r > 0.8), 
                            suggesting that players with larger hands tend to score high across all hand-related metrics.

                    - **Agility and Speed:** Lane Agility Time and 3_Quarter_Sprint show moderate correlations with other physical metrics, 
                            but less so with size-related features, highlighting their distinct measurement of quickness and speed.

                    - **Statistically Significant Differences:** Nearly all tested feature pairs with p-values < 0.05 indicate meaningful differences or associations, 
                            supporting the reliability of combine metrics for distinguishing player physical profiles.

                    - **Implications for Scouting:** The strongest statistical relationships reinforce the value of combine measurements in predicting player archetypes and potential NBA performance, 
                            especially for positions requiring size and athleticism.

                    These results validate the use of NBA Draft Combine data for player evaluation, modeling, and understanding how physical attributes translate to on-court success.


                """)

            with tab2:

                st.markdown("---")
                st.markdown("""
                    This section performs statistical tests to compare combine metrics across different player positions. 
                    We will conduct ANOVA and pairwise t-tests to assess the significance of differences between positions.
                """)
                stat_columns = data.drop(["Position", "Player", "Season"]).columns

                # Initialize session state if not already set
                if "stat_cat" not in st.session_state:
                    st.session_state.stat_cat = stat_columns[0]

                # Create selectboxes
                st.selectbox("Select Features", stat_columns, index=stat_columns.index(st.session_state.stat_cat), key="stat_cat")

                # Update session state manually
                stat_cat = st.session_state.stat_cat

                stat_tests_df = combine_stat_tests_by_position(df_wide_view_positions, stat_cat)
                if stat_tests_df is None:
                    st.markdown("No statistical tests performed. Please select a valid feature.")

                st.markdown(f"### Statistical Tests for {stat_cat} for all Positions")
                st.dataframe(stat_tests_df.select("Stat", "ANOVA_p")
                             .unique(["Stat"])
                             .filter(pl.col("ANOVA_p") < 0.05)
                             .sort("ANOVA_p")
                             .with_columns(
                                 pl.col("ANOVA_p").map_elements(lambda x: f"{x:.2e}")
                            )
                )
                st.markdown("---")
                st.markdown("#### Pairwise T-Tests")
                position_df = stat_tests_df.drop("ANOVA_p") \
                    .filter(pl.col("TTest_p") < 0.05) \
                    .sort("TTest_p") \
                    .with_columns(
                        (pl.col("Position1") + " vs " + pl.col("Position2")).alias("Position_Comparison")
                    )
                position_df = position_df.drop("Position1", "Position2").select("Stat", "Position_Comparison", "TTest_p")
                st.dataframe(position_df.pivot(
                    index="Position_Comparison",
                    values="TTest_p",
                    on="Stat"
                ).sort("Position_Comparison").fill_null(1.0)
                .with_columns(pl.col({stat_cat}).map_elements(lambda x: f"{x:.2e}"))
                )  # Fill nulls with 1.0 for no significant difference

                st.markdown("""

                ### Insights from ANOVA and t-tests on NBA Draft Combine Stat Comparisons

                Statistical tests reveal several important differences in combine metrics across NBA positions:

                - **Physical Attributes Vary Strongly by Position:** ANOVA and pairwise t-tests show extremely significant differences (p < 0.001) in Height, Standing Reach, Wingspan, and Weight between positions. 
                            Centers (C) and Power Forwards (PF) are consistently taller, heavier, and possess greater reach than Guards (PG, SG), confirming expected positional archetypes.

                - **Hand Measurements Distinguish Bigs from Guards:** Metrics like Length_x_Width_Hand, Width_of_Hand, and Length_of_Hand are significantly higher for bigs (C, PF) compared to guards, with p-values often < 1e-10. 
                            This suggests hand size is a notable differentiator by position.

                - **Athleticism and Speed Metrics:** Max Vertical Leap and Standing Vertical are higher for guards and wings (PG, SG, SF) than for bigs, indicating greater explosive leaping ability among perimeter players. 
                            Agility stats (Lane Agility Time, Shuttle Run, 3_Quarter_Sprint) also show significant differences, with guards generally posting better times.

                - **Body Composition:** Body Fat and BMI show moderate but significant differences, with guards tending to be leaner than forwards and centers.

                - **Statistically Robust Differences:** Nearly all combine metrics have at least one pairwise positional comparison with p < 0.05, indicating that physical and athletic traits are reliably distinct across NBA roles.

                **Conclusion:**  
                NBA Draft Combine data robustly differentiates player positions on physical, athletic, and body composition metrics. 
                            These differences are statistically significant and align with the demands of each role, supporting the use of combine stats for scouting and player evaluation.
                """)

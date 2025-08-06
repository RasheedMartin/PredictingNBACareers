import streamlit as st
import polars as pl
from pathlib import Path

st.title("Data Science Dashboard")

dir_path = Path("data/clean_data/NBA_Player_Data")

# Check if directory exists and list CSV files as strings
if not dir_path.exists():
    st.error(f"Directory not found: {dir_path}")
else:
    csv_files = [f.name for f in dir_path.iterdir() if f.is_file() and f.suffix == '.csv']

    if not csv_files:
        st.warning(f"No CSV files found in {dir_path}")
    else:
        selected_file = st.selectbox("Choose a CSV file", csv_files)

        # Load selected CSV
        filepath = dir_path / selected_file
        try:
            name = st.text_input("Your name")
            st.write(f"Hello, {name}!")
            # Testing:
            st.text("Simple text")
            st.markdown("**Bold** and *italic* with markdown")
            st.latex(r"E=mc^2")
            sidebar_option = st.sidebar.selectbox("Sidebar option", [1, 2, 3])
            st.write(f"Sidebar choice: {sidebar_option}")

            col1, col2 = st.columns(2)
            col1.write("Left column")
            col2.write("Right column")
            with st.expander("More info"):
                st.write("Hidden details here")
            st.header("Main Data")
            df = pl.read_csv(filepath)
            st.write(f"Preview of `{selected_file}`:")
            st.dataframe(df)
            st.header("Age vs Points")
            st.line_chart(df, x="Age", y="PTS")

            st.header("Slider for Time Series")
            if filepath.stem == "nba_list_no_agg":
                # Placeholder at the top
                df_placeholder = st.empty()  # Reserve a spot for dataframe
                years = df["Year"].unique()
                value = st.slider("Select a number", max_value=years.max(), min_value=years.min())  # min, max, default
                st.write(f"Value: {value}")
                df_placeholder.line_chart(df.filter((pl.col("Year") == value) & (pl.col("Team_Full_Name") != "Multiple Teams")).group_by("Team_Full_Name").agg(pl.col("PTS").mean()),
                              x="Team_Full_Name",
                              y="PTS"
                )
            with st.container():
                st.write("This is a sectionl")
                st.line_chart(df["G"])
                st.slider("This slider is in the same section", 0, 10)
            st.subheader("Summary Statistics")
            st.write(df.describe())

            if st.button("Run Model"):
                st.write("Button clicked!")
        except Exception as e:
            st.error(f"Error loading {selected_file}: {e}")

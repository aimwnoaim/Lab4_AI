# lab_eda_gui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")

# 2. Sidebar: Dataset Ingestion
st.sidebar.header("1. Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset (with basic validation)
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read the uploaded file. Please make sure it is a valid CSV. Error: {e}")
        st.stop()

    # 3. Dataset Overview
    st.subheader("Dataset Overview")

    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    st.write(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")

    st.write("**Column Data Types:**")
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Column", "Data Type"]
    st.dataframe(dtypes_df)

    # Missing value summary
    st.write("**Missing Values per Column:**")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.values
    })
    st.dataframe(missing_df)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats_df = numeric_df.agg(["mean", "median", "min", "max"]).T
        st.dataframe(stats_df)
    else:
        st.info("No numerical columns found in this dataset.")

    # 4. Attribute Selection
    st.sidebar.header("2. Attribute Selection")
    selected_column = st.sidebar.selectbox("Choose a column to visualize", df.columns)

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    st.sidebar.write(f"Detected type: **{column_type}**")

    # 5. Visualization Rendering
    st.subheader("Visualization")

    if column_type == "Numerical":
        # Histogram with seaborn
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        # Bar chart for categorical
        value_counts = df[selected_column].value_counts(dropna=False)
        percentages = (value_counts / value_counts.sum() * 100).round(1)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax, palette="viridis")
        ax.set_title(f"Frequency Counts of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        # Add percentage labels on top of bars
        for i, (count, pct) in enumerate(zip(value_counts.values, percentages.values)):
            ax.text(i, count, f"{pct}%", ha="center", va="bottom", fontsize=9)

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")

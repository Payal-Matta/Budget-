import streamlit as st
import pandas as pd

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Budget Dashboard 2014–2025",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Budget 2014-2025.csv")

df = load_data()

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.title("📊 Budget Dashboard (2014–2025)")
st.write("Interactive dashboard built with **no external libraries** so it works on GitHub/Streamlit Cloud without errors.")

st.divider()

# -------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------
st.sidebar.header("🔍 Filters")

years = sorted(df["Year"].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
if "Year" in numeric_cols:
    numeric_cols.remove("Year")

selected_col = st.sidebar.selectbox("Select Column", numeric_cols)

# -------------------------------------------------------------
# SUMMARY CARDS
# -------------------------------------------------------------
yr_value = df.loc[df["Year"] == selected_year, selected_col].values[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Year", selected_year)

with col2:
    st.metric(f"{selected_col}", f"{yr_value:,.2f}")

with col3:
    st.metric("Total Years", df["Year"].nunique())

st.divider()

# -------------------------------------------------------------
# LINE CHART
# -------------------------------------------------------------
st.subheader(f"📈 {selected_col} Trend Over Years")

line_df = df[["Year", selected_col]].set_index("Year")
st.line_chart(line_df, use_container_width=True)

# -------------------------------------------------------------
# BAR CHART
# -------------------------------------------------------------
st.subheader(f"📊 Year-wise Comparison: {selected_col}")

bar_df = df[["Year", selected_col]].set_index("Year")
st.bar_chart(bar_df, use_container_width=True)

# -------------------------------------------------------------
# DATA TABLE
# -------------------------------------------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df, use_container_width=True)

# -------------------------------------------------------------
# DOWNLOAD OPTION
# -------------------------------------------------------------
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv_data, "Budget_2014_2025.csv", "text/csv")

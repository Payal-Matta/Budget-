import streamlit as st
import pandas as pd
import plotly.express as px

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Budget Dashboard 2014-2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Custom CSS for UI/UX
# =======================
st.markdown("""
<style>
    .big-font {
        font-size: 26px !important;
        font-weight: 700;
        color: #2e7bcf;
    }
    .card {
        padding: 18px;
        background: #ffffff;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 6px solid #2e7bcf;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Load Data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("Budget 2014-2025.csv")
    return df

df = load_data()

st.markdown("<p class='big-font'>📊 Budget Analysis Dashboard (2014–2025)</p>", unsafe_allow_html=True)
st.write("A clean and interactive dashboard to explore yearly budget trends.")

# =======================
# Sidebar Filters
# =======================
st.sidebar.header("🔍 Filters")

# Year selection
years = sorted(df["Year"].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

# Column selector
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_column = st.sidebar.selectbox("Select Value Column", numeric_columns)

# =======================
# Summary Cards Section
# =======================
col1, col2, col3 = st.columns(3)

year_df = df[df["Year"] == selected_year]

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Selected Year", selected_year)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric(f"{selected_column} (Value)", f"{year_df[selected_column].iloc[0]:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Total Years in Data", len(df))
    st.markdown("</div>", unsafe_allow_html=True)

# =======================
# Charts Section
# =======================

st.subheader("📈 Trend Over the Years")

fig = px.line(
    df,
    x="Year",
    y=selected_column,
    markers=True,
    title=f"{selected_column} Trend (2014–2025)",
    color_discrete_sequence=['#2e7bcf']
)
fig.update_layout(
    template="plotly_white",
    title_x=0.2,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# =======================
# Bar Chart
# =======================
st.subheader("📊 Year-wise Comparison")

fig2 = px.bar(
    df,
    x="Year",
    y=selected_column,
    text=selected_column,
    title=f"{selected_column} Comparison Across Years",
    color="Year",
    color_continuous_scale="Blues"
)
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# =======================
# Data Table
# =======================
st.subheader("📋 Dataset Preview")
st.dataframe(df, use_container_width=True)

st.success("Dashboard Loaded Successfully 🚀")

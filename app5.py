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
# CUSTOM CSS FOR PREMIUM UI
# -------------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #dce9ff 0%, #f7f7ff 100%) !important;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #0066ff, #4f9cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: -20px;
}

.subtext {
    text-align:center;
    color:#5a5a5a;
    margin-top:-10px;
    font-size:16px;
}

.card {
    background: rgba(255, 255, 255, 0.45);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    transition: 0.2s ease-in-out;
    border: 1px solid rgba(255,255,255,0.5);
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.18);
}

.metric-label {
    font-size: 18px;
    color: #444;
    font-weight: 500;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #0057d9;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin: 20px 0 10px 0;
    color: #003f88;
}

hr.style {
    border: 0;
    height: 2px;
    background: linear-gradient(to right, #4f9cff, #0066ff);
    margin: 25px 0;
    border-radius: 50px;
}

</style>
""", unsafe_allow_html=True)

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
st.markdown("<div class='main-title'>📊 Budget Dashboard (2014–2025)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>A smooth, modern, glass-style dashboard with no external libraries</div>", unsafe_allow_html=True)

st.markdown("<hr class='style'>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("🔍 Filters")

years = sorted(df["Year"].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
if "Year" in numeric_cols:
    numeric_cols.remove("Year")

selected_col = st.sidebar.selectbox("Select Column", numeric_cols)

# -------------------------------------------------------------
# SUMMARY CARDS
# -------------------------------------------------------------
value_selected = df.loc[df["Year"] == selected_year, selected_col].values[0]

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Selected Year</div>
            <div class='metric-value'>{selected_year}</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>{selected_col}</div>
            <div class='metric-value'>{value_selected:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Total Years</div>
            <div class='metric-value'>{df['Year'].nunique()}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='style'>", unsafe_allow_html=True)

# -------------------------------------------------------------
# LINE CHART
# -------------------------------------------------------------
st.markdown("<div class='section-title'>📈 Yearly Trend</div>", unsafe_allow_html=True)

line_df = df[["Year", selected_col]].set_index("Year")
st.line_chart(line_df, use_container_width=True)

# -------------------------------------------------------------
# BAR CHART
# -------------------------------------------------------------
st.markdown("<div class='section-title'>📊 Year-wise Comparison</div>", unsafe_allow_html=True)

bar_df = df[["Year", selected_col]].set_index("Year")
st.bar_chart(bar_df, use_container_width=True)

st.markdown("<hr class='style'>", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA TABLE
# -------------------------------------------------------------
st.markdown("<div class='section-title'>📋 Dataset Preview</div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)

# DOWNLOAD BUTTON
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    csv,
    "Budget_2014_2025.csv",
    "text/csv",
)


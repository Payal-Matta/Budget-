# app.py  -- Streamlit dashboard without Plotly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============= Page config ============
st.set_page_config(
    page_title="Budget Dashboard 2014-2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CSS styling ============
st.markdown("""
<style>
    .big-title {font-size:28px; font-weight:700; color:#0b5ea8;}
    .subtle {color:#6b7280;}
    .card {padding:14px; border-radius:12px; background:#fff; box-shadow: 0 6px 18px rgba(0,0,0,0.04);}
</style>
""", unsafe_allow_html=True)

# ============= Load data ============
@st.cache_data
def load_data(path="Budget 2014-2025.csv"):
    df = pd.read_csv(path)
    # ensure Year column numeric
    if "Year" in df.columns:
        try:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
        except Exception:
            pass
    return df

try:
    df = load_data("Budget 2014-2025.csv")
except FileNotFoundError:
    st.error("CSV file not found. Make sure 'Budget 2014-2025.csv' is in the same folder as app.py.")
    st.stop()

# ============= Header ============
st.markdown(f"<div class='big-title'>📊 Budget Analysis Dashboard (2014–2025)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Interactive exploration — filters, summary cards, and charts (matplotlib)</div>", unsafe_allow_html=True)
st.markdown("---")

# ============= Sidebar filters ============
st.sidebar.header("Filters & Settings")
years = sorted(df["Year"].dropna().unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1 if years else 0)

# choose numeric column to visualize
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
# remove Year from numeric choices if present
if "Year" in numeric_cols:
    numeric_cols = [c for c in numeric_cols if c != "Year"]

if not numeric_cols:
    st.error("No numeric columns found in CSV to visualize.")
    st.stop()

selected_col = st.sidebar.selectbox("Value column", numeric_cols, index=0)

# optional smoothing
smooth = st.sidebar.checkbox("Show moving average (3-year)", value=True)

# ============= Summary cards ============
col1, col2, col3 = st.columns([1.2,1.2,1.2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Selected Year", value=str(selected_year))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    val = df.loc[df["Year"] == selected_year, selected_col]
    val_text = f"{val.values[0]:,.2f}" if len(val) > 0 and pd.notna(val.values[0]) else "N/A"
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric(f"{selected_col}", value=val_text)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    total_years = df["Year"].nunique()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Years in dataset", value=str(total_years))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============= Charts (matplotlib) ============
st.subheader("Trend across Years")

plot_df = df.sort_values("Year").dropna(subset=[selected_col, "Year"])
x = plot_df["Year"].values
y = plot_df[selected_col].values

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(x, y, marker='o', linewidth=2)
if smooth and len(y) >= 3:
    ma = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values
    ax.plot(x, ma, linestyle='--', linewidth=1.6)

ax.set_title(f"{selected_col} Trend (2014-2025)", fontsize=12, weight='bold')
ax.set_xlabel("Year")
ax.set_ylabel(selected_col)
ax.grid(alpha=0.15)
# format y axis with commas
ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:,.0f}"))
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.subheader("Year-wise Bar Comparison")
fig2, ax2 = plt.subplots(figsize=(10, 4.2))
bars = ax2.bar(plot_df["Year"].astype(str), plot_df[selected_col])
ax2.set_title(f"{selected_col} by Year", fontsize=12, weight='bold')
ax2.set_xlabel("Year")
ax2.set_ylabel(selected_col)
ax2.bar_label(bars, labels=[f"{v:,.0f}" for v in plot_df[selected_col].values], padding=3, fontsize=9)
ax2.grid(axis='y', alpha=0.12)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)

# ============= Data table & download ============
st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

# CSV download
csv = df.to_csv_

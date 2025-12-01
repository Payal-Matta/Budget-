# streamlit_dashboard_budget.py
# Interactive Streamlit dashboard tailored for the uploaded file: /mnt/data/Budget 2014-2025.csv
# Features:
# - Data preview & cleaning
# - Summary statistics and distributions
# - Time-series and comparison charts (Plotly)
# - Year-wise treemap for budget head breakdown
# - Correlation heatmap
# - KMeans clustering (interactive) + PCA scatter
# - Simple forecasting (Linear Regression) for a selected budget head
# - Export cleaned/filtered dataset

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Budget Dashboard (2014-2025)", layout="wide")

# ---- Helpers ----
@st.cache_data
def load_data(path="/mnt/data/Budget 2014-2025.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_data
def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

@st.cache_data
def preprocess_for_clustering(df, features):
    sub = df[features].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub)
    return sub, scaled, scaler

# ---- Load data ----
st.title("📊 Budget Dashboard — 2014 to 2025")
st.markdown("Upload a CSV or use the detected file. This app is tailored to the uploaded file `/mnt/data/Budget 2014-2025.csv`.")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"]) 
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_data()

st.sidebar.markdown("---")

# ---- Quick data preview and cleaning ----
st.header("1. Data preview & cleaning")
with st.expander("Preview dataset"):
    st.dataframe(df.head(20))

st.write("**Shape:**", df.shape)

# Attempt to detect a 'Year' column or similar
possible_year_cols = [c for c in df.columns if 'year' in c.lower() or 'yr' in c.lower()]
if len(possible_year_cols) > 0:
    year_col = possible_year_cols[0]
else:
    # try to find integer column with values in range 2000-2030
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    year_col = None
    for c in int_cols:
        vals = df[c].dropna().unique()
        if len(vals)>0 and ((vals >= 2000) & (vals <= 2030)).any():
            year_col = c
            break

if year_col is None:
    st.warning("No explicit 'Year' column detected. Some time-series features will be limited. You can still pick numeric columns for analysis.")
else:
    st.success(f"Detected year column: **{year_col}**")

# Allow user to rename year column if needed
if year_col is not None:
    with st.expander("Year column settings"):
        year_col = st.selectbox("Choose Year column for time-series analyses", options=[None] + list(df.columns), index=(1 if year_col in df.columns else 0))

# Fill simple NaNs optionally
if st.sidebar.checkbox("Fill numeric NaNs with 0 (quick)", value=False):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

# ---- Summary statistics ----
st.header("2. Summary statistics & distributions")
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Descriptive statistics (numeric)")
    st.dataframe(df.describe().T)
with col2:
    st.subheader("Missing values")
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss>0])

# Choose numeric column(s) for visualization
num_cols = numeric_columns(df)
if len(num_cols) == 0:
    st.error("No numeric columns found for plotting and clustering.")
    st.stop()

selected_cols = st.multiselect("Select numeric column(s) to visualize", options=num_cols, default=num_cols[:2])

# ---- Time-series and comparison charts ----
st.header("3. Time-series & comparison")
if year_col is not None:
    years = sorted(df[year_col].dropna().unique())
    sel_years = st.slider("Select year range", min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))
    mask = (df[year_col] >= sel_years[0]) & (df[year_col] <= sel_years[1])
    df_time = df.loc[mask]

    if len(selected_cols) > 0:
        st.subheader("Line / area chart over years")
        # aggregate by year if dataset is granular
        agg = df_time.groupby(year_col)[selected_cols].sum().reset_index()
        fig = px.line(agg, x=year_col, y=selected_cols, title="Time-series of selected budget heads")
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Stacked area (relative share)")
        fig2 = go.Figure()
        for c in selected_cols:
            fig2.add_trace(go.Scatter(x=agg[year_col], y=agg[c], stackgroup='one', name=c))
        fig2.update_layout(title="Stacked area chart by year", xaxis_title=year_col)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Pick at least one numeric column from the selector to plot time-series.")
else:
    st.info("No year column selected/detected — cannot produce time-series charts.")

# ---- Year-wise treemap for breakdown ----
st.header("4. Year-wise breakdown (Treemap)")
with st.expander("Treemap options"):
    treemap_year = st.selectbox("Select year for breakdown", options=[None] + (sorted(df[year_col].unique().tolist()) if year_col is not None else []))
    label_col = st.selectbox("Choose label/category column (e.g., Budget Head)", options=[None] + list(df.columns))
    value_col = st.selectbox("Choose value column (numeric)", options=[None] + num_cols)

if treemap_year and label_col and value_col:
    df_year = df[df[year_col]==treemap_year]
    treedata = df_year.groupby(label_col)[value_col].sum().reset_index()
    fig = px.treemap(treedata, path=[label_col], values=value_col, title=f"Budget breakdown — {treemap_year}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choose a year, a label column and a numeric value column to view the treemap.")

# ---- Correlation heatmap ----
st.header("5. Correlation heatmap")
corr = df[num_cols].corr()
fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation matrix for numeric columns')
st.plotly_chart(fig, use_container_width=True)

# ---- KMeans clustering + PCA ----
st.header("6. KMeans clustering (interactive)")
with st.expander("Clustering options"):
    clustering_features = st.multiselect("Choose numeric features for clustering (2-8 recommended)", options=num_cols, default=num_cols[:3])
    n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=3)
    run_cluster = st.button("Run clustering")

if run_cluster:
    if len(clustering_features) < 2:
        st.error("Select at least 2 numeric features for clustering.")
    else:
        sub, scaled, scaler = preprocess_for_clustering(df, clustering_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled)
        sub2 = sub.copy()
        sub2['cluster'] = labels.astype(str)

        st.write("Cluster sizes:")
        st.dataframe(pd.Series(labels).value_counts().sort_index())

        # PCA for 2D scatter
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        scatter_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
        scatter_df['cluster'] = labels.astype(str)
        hover_cols = clustering_features if len(clustering_features)<=4 else clustering_features[:4]
        fig = px.scatter(scatter_df, x='PC1', y='PC2', color='cluster', title='PCA projection of clusters', hover_data=hover_cols)
        st.plotly_chart(fig, use_container_width=True)

        # Show cluster centers (inverse transform)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=clustering_features)
        centers_df.index.name = 'cluster'
        st.subheader('Cluster centers (approx)')
        st.dataframe(centers_df)

# ---- Simple forecasting for 1 budget head ----
st.header("7. Simple forecasting (Linear Regression)")
with st.expander("Forecasting options"):
    forecast_col = st.selectbox("Choose numeric column to forecast", options=[None] + num_cols)
    if year_col is not None:
        train_years = st.slider("Use data up to year (inclusive) for training — select max year for training", min_value=int(min(df[year_col].dropna().unique())), max_value=int(max(df[year_col].dropna().unique())), value=int(max(df[year_col].dropna().unique())))
    else:
        train_years = st.number_input("Train up to year (if available)", value=2021)
    forecast_ahead = st.number_input("Forecast how many years ahead (integer)", min_value=1, max_value=10, value=2)
    run_forecast = st.button("Run forecast")

if run_forecast:
    if not year_col:
        st.error("Forecasting requires a detected Year column.")
    elif not forecast_col:
        st.error("Pick a numeric column to forecast.")
    else:
        dff = df[[year_col, forecast_col]].dropna().groupby(year_col)[forecast_col].sum().reset_index()
        df_train = dff[dff[year_col] <= train_years]
        X = df_train[[year_col]].values
        y = df_train[forecast_col].values
        model = LinearRegression()
        model.fit(X, y)
        last_year = int(dff[year_col].max())
        future_years = np.arange(last_year+1, last_year+1+forecast_ahead)
        y_pred_future = model.predict(future_years.reshape(-1,1))

        # Build chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train[year_col], y=df_train[forecast_col], mode='markers+lines', name='Train'))
        if (dff[year_col] > train_years).any():
            df_test = dff[dff[year_col] > train_years]
            fig.add_trace(go.Scatter(x=df_test[year_col], y=df_test[forecast_col], mode='markers+lines', name='Actual (holdout)'))
        fig.add_trace(go.Scatter(x=future_years, y=y_pred_future, mode='markers+lines', name='Forecast'))
        fig.update_layout(title=f'Forecast for {forecast_col}', xaxis_title=year_col, yaxis_title=forecast_col)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics on holdout (if available)
        if (dff[year_col] > train_years).any():
            df_test = dff[dff[year_col] > train_years]
            y_test = df_test[forecast_col].values
            X_test = df_test[[year_col]].values
            y_pred_test = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred_test, squared=False)
            st.write(f"Holdout RMSE: {rmse:.2f}")

# ---- Export cleaned/filtered data ----
st.header("8. Export data")
with st.expander("Download filtered / cleaned dataset"):
    if year_col is not None and 'df_time' in locals():
        export_df = df_time.copy()
    else:
        export_df = df.copy()
    st.dataframe(export_df.head(10))
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='budget_filtered.csv', mime='text/csv')

# ---- Footer / notes ----
st.markdown("---")
st.write("**Notes:** This dashboard is built as a general, interactive explorer for the uploaded Budget CSV. Feel free to modify feature lists, add domain-specific transformations (e.g., grouping budget heads), or enhance forecasting with richer time-series models.")

# End of file

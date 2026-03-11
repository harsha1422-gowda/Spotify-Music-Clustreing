import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os

st.set_page_config(
    page_title="Spotify Music Clustering",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Music Clustering Dashboard")
st.markdown("Discover music patterns using audio features")

# Sidebar
st.sidebar.header("Settings")

clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)

st.sidebar.markdown("---")
st.sidebar.write("Dataset Source: Kaggle")

# Load dataset
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    data = pd.read_csv(os.path.join(path, "SpotifyFeatures.csv"))
    return data

data = load_data()

# Select features
features = ['danceability','energy','tempo','loudness','valence']
data = data[features].dropna()

# Normalize features
data_norm = (data - data.min()) / (data.max() - data.min())

# Simple clustering
data_norm["Cluster"] = pd.qcut(data_norm["energy"], q=clusters, labels=False)

# Metrics row
col1, col2, col3 = st.columns(3)

col1.metric("Songs Analysed", len(data_norm))
col2.metric("Features Used", len(features))
col3.metric("Clusters Created", clusters)

st.markdown("---")

# Charts section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Distribution")
    st.bar_chart(data_norm["Cluster"].value_counts())

with col2:
    st.subheader("Feature Comparison")
    st.line_chart(data_norm.groupby("Cluster").mean())

st.markdown("---")

# Dataset preview
st.subheader("Dataset Preview")

st.dataframe(data_norm.head(100), use_container_width=True)

# Song explorer
st.markdown("---")
st.subheader("🎧 Feature Explorer")

feature_choice = st.selectbox(
    "Select feature to explore",
    features
)

st.line_chart(data_norm[feature_choice])

st.markdown("---")

st.caption("Spotify clustering demo built with Streamlit")

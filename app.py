import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import kagglehub
import os

st.title("Spotify Song Clustering")

# Download dataset from Kaggle
path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")

data = pd.read_csv(os.path.join(path, "SpotifyFeatures.csv"))

# Select features
features = ['danceability','energy','tempo','loudness','valence']

# Create cleaned dataset
data_clean = data[features].dropna().copy()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_clean)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
data_clean["Cluster"] = clusters

st.subheader("Sample Clustered Songs")
st.dataframe(data_clean.head())

# Cluster distribution
st.subheader("Cluster Distribution")
st.bar_chart(data_clean["Cluster"].value_counts())

# Show cluster statistics
st.subheader("Cluster Feature Averages")
st.dataframe(data_clean.groupby("Cluster").mean())

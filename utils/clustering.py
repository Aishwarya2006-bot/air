# utils/clustering.py

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st


# =====================================================
# NUMERIC FEATURE EXTRACTION
# =====================================================

def get_numeric_features(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
        .copy()
    )

    numeric_df = numeric_df.dropna()

    return numeric_df


# =====================================================
# CLUSTER LABELS
# =====================================================

def generate_cluster_labels(
    cluster_profiles
):

    labels = {}

    for cluster_id in cluster_profiles.index:

        labels[cluster_id] = (
            f"Environmental Pattern {cluster_id}"
        )

    return labels


# =====================================================
# PATTERN RECOGNITION PIPELINE
# =====================================================

@st.cache_data
def build_pattern_recognition_pipeline(df, n_clusters=4):
    """
    Builds a clustering pipeline that handles missing values, scales features,
    and runs KMeans clustering.
    """
    # 1. Select only numeric columns for clustering, dropping 'Cluster' if it exists
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if 'Cluster' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Cluster'])
        
    features = numeric_df.columns.tolist()
    
    # 2. Impute missing values (NaNs) with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numeric_df)
    
    # 3. Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    
    # 4. Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 5. Create output copies to avoid modifying original source dataframes
    clustered_data = df.copy()
    clustered_data['Cluster'] = cluster_labels
    
    # 6. Calculate cluster profiles (means) using the imputed data structure
    imputed_df = pd.DataFrame(imputed_data, columns=features)
    imputed_df['Cluster'] = cluster_labels
    cluster_profiles = imputed_df.groupby('Cluster').mean()
    
    # 7. Mock an anomaly score dataframe for your app interface consistency
    anomaly_data = clustered_data.copy()
    # Calculate distance to cluster centers as an basic anomaly score
    distances = kmeans.transform(scaled_data)
    anomaly_data['anomaly_score'] = distances.min(axis=1)
    # Mark the top 5% highest distances as anomalies
    threshold = np.percentile(anomaly_data['anomaly_score'], 95)
    anomaly_data['is_anomaly'] = anomaly_data['anomaly_score'] > threshold
    
    return {
        "clustered_data": clustered_data,
        "cluster_profiles": cluster_profiles,
        "anomaly_data": anomaly_data
    }
    # =====================================
    # ANOMALY DETECTION
    # =====================================

    isolation_model = (
        IsolationForest(
            contamination=0.05,
            random_state=42
        )
    )

    anomaly_flags = (
        isolation_model.fit_predict(
            scaled_data
        )
    )

    anomaly_scores = (
        isolation_model.score_samples(
            scaled_data
        )
    )

    anomaly_df = (
        clustered_df.copy()
    )

    anomaly_df[
        "Anomaly"
    ] = anomaly_flags

    anomaly_df[
        "Anomaly_Score"
    ] = anomaly_scores

    return {

        "clustered_data":
            clustered_df,

        "cluster_profiles":
            cluster_profiles,

        "anomaly_data":
            anomaly_df,

        "kmeans_model":
            kmeans,

        "scaler":
            scaler
    }


# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================

def calculate_cluster_distribution(
    clustered_df
):

    if clustered_df.empty:

        return pd.DataFrame()

    counts = (
        clustered_df[
            "Cluster"
        ]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Count"
    ]

    counts[
        "Percentage"
    ] = (
        counts["Count"]
        /
        counts["Count"].sum()
        * 100
    ).round(2)

    return counts


# =====================================================
# TOP ANOMALIES
# =====================================================

def get_top_anomalies(
    anomaly_df,
    n=15
):

    if anomaly_df.empty:

        return pd.DataFrame()

    anomalies = (
        anomaly_df[
            anomaly_df[
                "Anomaly"
            ] == -1
        ]
        .copy()
    )

    anomalies = (
        anomalies.sort_values(
            "Anomaly_Score"
        )
    )

    return anomalies.head(n)


# =====================================================
# ENVIRONMENTAL EVENT DETECTOR
# =====================================================

def environmental_event_detector(
    numeric_df,
    z_threshold=2.5
):

    if numeric_df.empty:

        return pd.DataFrame()

    numeric_df = (
        numeric_df.copy()
    )

    z_scores = np.abs(

        (
            numeric_df
            -
            numeric_df.mean()
        )

        /

        numeric_df.std()
    )

    mask = (
        z_scores > z_threshold
    )

    event_rows = (
        mask.any(axis=1)
    )

    events = (
        numeric_df[
            event_rows
        ]
        .copy()
    )

    return events


# =====================================================
# CLUSTER SUMMARY
# =====================================================

def cluster_summary(
    clustered_df
):

    if clustered_df.empty:

        return pd.DataFrame()

    summary = (

        clustered_df

        .groupby(
            "Cluster"
        )

        .agg(
            Count=(
                "Cluster",
                "size"
            )
        )

        .reset_index()
    )

    return summary

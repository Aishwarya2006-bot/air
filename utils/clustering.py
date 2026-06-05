# utils/clustering.py

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
def build_pattern_recognition_pipeline(
    df,
    n_clusters=4
):

    numeric_df = (
        get_numeric_features(df)
    )

    if numeric_df.empty:

        raise ValueError(
            "No numeric features available for clustering."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        scaled_data
    )

    clustered_df = (
        df.loc[
            numeric_df.index
        ]
        .copy()
    )

    clustered_df[
        "Cluster"
    ] = clusters

    cluster_profiles = (
        clustered_df
        .groupby(
            "Cluster"
        )
        .mean(
            numeric_only=True
        )
    )

    cluster_labels = (
        generate_cluster_labels(
            cluster_profiles
        )
    )

    clustered_df[
        "Cluster_Label"
    ] = clustered_df[
        "Cluster"
    ].map(
        cluster_labels
    )

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

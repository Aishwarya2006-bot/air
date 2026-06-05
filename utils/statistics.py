# utils/statistics.py

import pandas as pd
import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr
)

import streamlit as st


# =====================================================
# NUMERIC DATA EXTRACTION
# =====================================================

def get_numeric_dataframe(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
        .copy()
    )

    return numeric_df


# =====================================================
# CORRELATION MATRIX
# =====================================================

@st.cache_data
def calculate_correlation_matrix(df):

    numeric_df = get_numeric_dataframe(df)

    if numeric_df.empty:

        return pd.DataFrame()

    return numeric_df.corr()


# =====================================================
# PEARSON ANALYSIS
# =====================================================

def pearson_analysis(
    df,
    x_col,
    y_col
):

    try:

        temp = (
            df[[x_col, y_col]]
            .copy()
        )

        temp[x_col] = pd.to_numeric(
            temp[x_col],
            errors="coerce"
        )

        temp[y_col] = pd.to_numeric(
            temp[y_col],
            errors="coerce"
        )

        temp = temp.dropna()

        if len(temp) < 3:

            return None

        corr, pval = pearsonr(
            temp[x_col],
            temp[y_col]
        )

        return {
            "correlation": corr,
            "p_value": pval
        }

    except Exception:

        return None


# =====================================================
# SPEARMAN ANALYSIS
# =====================================================

def spearman_analysis(
    df,
    x_col,
    y_col
):

    try:

        temp = (
            df[[x_col, y_col]]
            .copy()
        )

        temp[x_col] = pd.to_numeric(
            temp[x_col],
            errors="coerce"
        )

        temp[y_col] = pd.to_numeric(
            temp[y_col],
            errors="coerce"
        )

        temp = temp.dropna()

        if len(temp) < 3:

            return None

        corr, pval = spearmanr(
            temp[x_col],
            temp[y_col]
        )

        return {
            "correlation": corr,
            "p_value": pval
        }

    except Exception:

        return None


# =====================================================
# WEATHER SIGNIFICANCE
# =====================================================

def weather_pollution_significance(
    df,
    weather_columns,
    pollution_columns
):

    results = []

    for weather in weather_columns:

        for pollution in pollution_columns:

            try:

                result = pearson_analysis(
                    df,
                    weather,
                    pollution
                )

                if result is None:
                    continue

                significance = (
                    "Statistically Significant"
                    if result["p_value"] < 0.05
                    else "Not Statistically Significant"
                )

                results.append(
                    {
                        "Weather Variable":
                            weather,
                        "Pollution Variable":
                            pollution,
                        "Pearson_r":
                            round(
                                result["correlation"],
                                4
                            ),
                        "P_Value":
                            round(
                                result["p_value"],
                                6
                            ),
                        "Significance":
                            significance
                    }
                )

            except Exception:
                continue

    return pd.DataFrame(
        results
    )


# =====================================================
# LAG ANALYSIS
# =====================================================

def lag_correlation_analysis(
    df,
    source_columns,
    target_columns,
    lags=[1, 2]
):

    results = []

    for source in source_columns:

        for target in target_columns:

            try:

                base = (
                    df[[source, target]]
                    .copy()
                )

                base[source] = pd.to_numeric(
                    base[source],
                    errors="coerce"
                )

                base[target] = pd.to_numeric(
                    base[target],
                    errors="coerce"
                )

                for lag in lags:

                    lag_col = (
                        f"{source}_lag_{lag}"
                    )

                    base[lag_col] = (
                        base[source]
                        .shift(lag)
                    )

                    lag_df = (
                        base[
                            [lag_col, target]
                        ]
                        .dropna()
                    )

                    if len(lag_df) < 3:
                        continue

                    corr, pval = pearsonr(
                        lag_df[lag_col],
                        lag_df[target]
                    )

                    results.append(
                        {
                            "Source":
                                source,
                            "Target":
                                target,
                            "Lag":
                                lag,
                            "Correlation":
                                round(
                                    corr,
                                    4
                                ),
                            "P_Value":
                                round(
                                    pval,
                                    6
                                )
                        }
                    )

            except Exception:
                continue

    return pd.DataFrame(
        results
    )


# =====================================================
# KPI GENERATION
# =====================================================

@st.cache_data
def calculate_basic_kpis(df):

    numeric_df = get_numeric_dataframe(df)

    kpis = {}

    for col in numeric_df.columns:

        try:

            kpis[col] = {

                "mean":
                    float(
                        numeric_df[col]
                        .mean()
                    ),

                "median":
                    float(
                        numeric_df[col]
                        .median()
                    ),

                "min":
                    float(
                        numeric_df[col]
                        .min()
                    ),

                "max":
                    float(
                        numeric_df[col]
                        .max()
                    )
            }

        except Exception:
            continue

    return kpis


# =====================================================
# STRONGEST CORRELATIONS
# =====================================================

def strongest_correlations(
    df,
    top_n=20
):

    numeric_df = get_numeric_dataframe(df)

    if numeric_df.empty:

        return pd.DataFrame()

    corr_matrix = (
        numeric_df.corr()
        .abs()
    )

    corr_pairs = []

    cols = corr_matrix.columns

    for i in range(len(cols)):

        for j in range(i + 1, len(cols)):

            corr_pairs.append(
                {
                    "Variable_1":
                        cols[i],
                    "Variable_2":
                        cols[j],
                    "Correlation":
                        corr_matrix.iloc[
                            i,
                            j
                        ]
                }
            )

    result = pd.DataFrame(
        corr_pairs
    )

    result = (
        result.sort_values(
            "Correlation",
            ascending=False
        )
        .head(top_n)
    )

    return result.reset_index(
        drop=True
    )

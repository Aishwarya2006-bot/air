import pandas as pd
import numpy as np
import streamlit as st


# =====================================================
# CSV LOADER
# =====================================================

@st.cache_data
def load_csv(uploaded_file):
    """
    Load CSV safely.
    """

    try:

        df = pd.read_csv(uploaded_file)

        return df

    except Exception as e:

        raise ValueError(
            f"Unable to read CSV file: {e}"
        )


# =====================================================
# DATASET VALIDATION
# =====================================================

def validate_wind_dataset(df):

    if "date" not in df.columns:

        raise KeyError(
            "Wind dataset must contain column: date"
        )

    return True


def validate_no2_dataset(df):

    if "date" not in df.columns:

        raise KeyError(
            "NO₂ dataset must contain column: date"
        )

    return True


def validate_lst_dataset(df):

    if "date" not in df.columns:

        raise KeyError(
            "LST dataset must contain column: date"
        )

    return True


# =====================================================
# CLEAN OBJECT COLUMNS
# =====================================================

def clean_object_columns(df):

    df = df.copy()

    for col in df.columns:

        if col.lower() in ["date", "timestamp"]:
            continue

        if df[col].dtype == "object":

            cleaned = (
                df[col]
                .astype(str)
                .str.replace("micromol/m2", "", regex=False)
                .str.replace("µmol/m2", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            converted = pd.to_numeric(
                cleaned,
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                df[col] = converted

    return df


# =====================================================
# PREPROCESS DATAFRAME
# =====================================================

def preprocess_dataframe(df):

    df = df.copy()

    if "date" not in df.columns:

        raise KeyError(
            "Column 'date' is required."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    df = df.drop_duplicates(
        subset=["date"]
    )

    df = df.sort_values(
        "date"
    )

    df = clean_object_columns(
        df
    )

    return df


# =====================================================
# DATA MERGE
# =====================================================

@st.cache_data
def merge_datasets(
    wind_df,
    no2_df,
    lst_df
):

    merged = pd.merge(
        wind_df,
        no2_df,
        on="date",
        how="inner"
    )

    merged = pd.merge(
        merged,
        lst_df,
        on="date",
        how="inner"
    )

    merged = merged.sort_values(
        "date"
    )

    merged = merged.ffill().bfill()

    return merged


# =====================================================
# SAMPLE DATA GENERATOR
# =====================================================

@st.cache_data
def generate_sample_datasets():

    np.random.seed(42)

    dates = pd.date_range(
        start="2023-01-01",
        periods=365,
        freq="D"
    )

    wind = np.random.normal(
        12,
        3,
        len(dates)
    )

    lst = (
        30
        + np.sin(
            np.linspace(
                0,
                10,
                len(dates)
            )
        ) * 5
        + np.random.normal(
            0,
            1,
            len(dates)
        )
    )

    no2 = (
        50
        - wind * 0.8
        + lst * 0.6
        + np.random.normal(
            0,
            3,
            len(dates)
        )
    )

    wind_df = pd.DataFrame({
        "date": dates,
        "Wind_Speed": wind
    })

    no2_df = pd.DataFrame({
        "date": dates,
        "NO2_Level": no2
    })

    lst_df = pd.DataFrame({
        "date": dates,
        "LST": lst
    })

    return (
        wind_df,
        no2_df,
        lst_df
    )


# =====================================================
# COLUMN IDENTIFICATION
# =====================================================

def identify_pollution_columns(df):

    keywords = [
        "no2",
        "pollution",
        "pm",
        "aqi"
    ]

    cols = []

    for col in df.columns:

        name = col.lower()

        if any(
            key in name
            for key in keywords
        ):
            cols.append(col)

    return cols


def identify_wind_columns(df):

    keywords = [
        "wind",
        "speed"
    ]

    cols = []

    for col in df.columns:

        name = col.lower()

        if any(
            key in name
            for key in keywords
        ):
            cols.append(col)

    return cols


def identify_temperature_columns(df):

    keywords = [
        "lst",
        "temp",
        "temperature"
    ]

    cols = []

    for col in df.columns:

        name = col.lower()

        if any(
            key in name
            for key in keywords
        ):
            cols.append(col)

    return cols


# =====================================================
# GET NUMERIC COLUMNS
# =====================================================

def get_numeric_columns(df):
    """
    Get all numeric columns from the dataframe.
    """
    
    numeric_cols = (
        df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
    )
    
    return numeric_cols

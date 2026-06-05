# utils/forecasting.py

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import streamlit as st


# =====================================================
# PREPARE FEATURES
# =====================================================

def prepare_forecasting_dataset(
    df,
    target_column
):

    working_df = df.copy()

    numeric_df = (
        working_df.select_dtypes(
            include=np.number
        )
        .copy()
    )

    if target_column not in numeric_df.columns:

        raise ValueError(
            f"{target_column} is not numeric."
        )

    # ==========================
    # LAG FEATURES
    # ==========================

    numeric_df[
        f"{target_column}_lag1"
    ] = (
        numeric_df[
            target_column
        ]
        .shift(1)
    )

    numeric_df[
        f"{target_column}_lag2"
    ] = (
        numeric_df[
            target_column
        ]
        .shift(2)
    )

    numeric_df = (
        numeric_df
        .ffill()
        .bfill()
    )

    return numeric_df


# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_forecasting_model(
    df,
    target_column
):

    data = (
        prepare_forecasting_dataset(
            df,
            target_column
        )
    )

    X = (
        data.drop(
            columns=[
                target_column
            ]
        )
    )

    y = data[
        target_column
    ]

    X = (
        X.fillna(
            X.median()
        )
    )

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )
    )

    model = (
        RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            random_state=42,

            n_jobs=-1
        )
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    r2 = (
        r2_score(
            y_test,
            predictions
        )
    )

    mae = (
        mean_absolute_error(
            y_test,
            predictions
        )
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    prediction_df = pd.DataFrame({

        "Actual":
            y_test.values,

        "Predicted":
            predictions
    })

    metrics = {

        "R2":
            round(
                r2,
                4
            ),

        "MAE":
            round(
                mae,
                4
            ),

        "RMSE":
            round(
                rmse,
                4
            )
    }

    return {

        "model":
            model,

        "metrics":
            metrics,

        "prediction_df":
            prediction_df,

        "feature_columns":
            list(
                X.columns
            ),

        "X":
            X,

        "y":
            y
    }


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def get_feature_importance(
    model,
    feature_columns
):

    if model is None:

        return pd.DataFrame()

    importance_df = pd.DataFrame({

        "Feature":
            feature_columns,

        "Importance":
            model.feature_importances_
    })

    importance_df = (

        importance_df

        .sort_values(
            "Importance",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )

    return importance_df


# =====================================================
# FUTURE FORECAST
# =====================================================

def forecast_future_values(
    df,
    target_column,
    periods=30
):

    trained = (
        train_forecasting_model(
            df,
            target_column
        )
    )

    model = trained["model"]

    data = (
        prepare_forecasting_dataset(
            df,
            target_column
        )
    )

    feature_columns = [

        col

        for col in data.columns

        if col != target_column
    ]

    latest_row = (
        data[
            feature_columns
        ]
        .iloc[-1]
        .copy()
    )

    future_predictions = []

    current_row = (
        latest_row.copy()
    )

    for _ in range(periods):

        prediction = (

            model.predict(

                pd.DataFrame(
                    [current_row]
                )
            )[0]
        )

        future_predictions.append(
            prediction
        )

    if "date" in df.columns:

        last_date = (
            pd.to_datetime(
                df["date"]
            ).max()
        )

    else:

        last_date = pd.Timestamp.today()

    future_dates = pd.date_range(

        start=last_date
        + pd.Timedelta(days=1),

        periods=periods,

        freq="D"
    )

    forecast_df = pd.DataFrame({

        "Date":
            future_dates,

        "Forecast":
            future_predictions
    })

    return forecast_df


# =====================================================
# MODEL SUMMARY
# =====================================================

def model_summary(
    metrics
):

    summary = pd.DataFrame({

        "Metric":
            [
                "R²",
                "MAE",
                "RMSE"
            ],

        "Value":
            [
                metrics["R2"],
                metrics["MAE"],
                metrics["RMSE"]
            ]
    })

    return summary

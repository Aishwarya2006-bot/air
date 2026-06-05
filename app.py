# app.py (Complete - Error-Proof Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import (
    load_csv,
    validate_wind_dataset,
    validate_no2_dataset,
    validate_lst_dataset,
    preprocess_dataframe,
    merge_datasets,
    generate_sample_datasets,
    identify_pollution_columns,
    identify_wind_columns,
    identify_temperature_columns,
    get_numeric_columns
)

from utils.statistics import (
    calculate_correlation_matrix,
    pearson_analysis,
    spearman_analysis,
    weather_pollution_significance,
    lag_correlation_analysis,
    calculate_basic_kpis,
    strongest_correlations
)

from utils.clustering import (
    build_pattern_recognition_pipeline,
    calculate_cluster_distribution,
    environmental_event_detector,
    get_top_anomalies
)

from utils.forecasting import (
    train_forecasting_model,
    forecast_future_values,
    get_feature_importance
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Urban Air Quality Correlation Engine",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# SAFE GLOBAL VARIABLES
# =====================================================

merged_df = None
clustered_df = None
pattern_results = None
prediction_df = None
model = None
forecast_df = None
metrics = None

# =====================================================
# TITLE
# =====================================================

st.title("🌍 Urban Air Quality Correlation Engine")

st.markdown(
    """
Advanced Environmental Intelligence Platform

Features:
- Multi-File Data Fusion
- Correlation Analytics
- Lag Analysis
- Significance Testing
- Pattern Recognition
- Anomaly Detection
- Predictive Forecasting
"""
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📂 Upload Datasets")
wind_file = st.sidebar.file_uploader("Wind Dataset", type=["csv"])
no2_file = st.sidebar.file_uploader("NO₂ Dataset", type=["csv"])
lst_file = st.sidebar.file_uploader("LST Dataset", type=["csv"])

use_sample_data = False
if (wind_file is None or no2_file is None or lst_file is None):
    st.sidebar.warning("Upload all 3 datasets or generate sample datasets.")
    if st.sidebar.button("Generate & Load 3 Sample Interrelated Datasets"):
        use_sample_data = True

# =====================================================
# LOAD DATA
# =====================================================

try:
    if use_sample_data:
        wind_df, no2_df, lst_df = generate_sample_datasets()
        merged_df = merge_datasets(wind_df, no2_df, lst_df)
        st.success("Sample datasets loaded successfully.")
    elif (wind_file is not None and no2_file is not None and lst_file is not None):
        wind_df = load_csv(wind_file)
        no2_df = load_csv(no2_file)
        lst_df = load_csv(lst_file)
        
        validate_wind_dataset(wind_df)
        validate_no2_dataset(no2_df)
        validate_lst_dataset(lst_df)
        
        wind_df = preprocess_dataframe(wind_df)
        no2_df = preprocess_dataframe(no2_df)
        lst_df = preprocess_dataframe(lst_df)
        
        merged_df = merge_datasets(wind_df, no2_df, lst_df)
except KeyError as e:
    st.error(f"Column Error: {e}")
except ValueError as e:
    st.error(f"Data Error: {e}")
except Exception as e:
    st.error(f"Unexpected Error: {e}")

# =====================================================
# STOP IF NO DATA
# =====================================================

if merged_df is None:
    st.info("Upload all three datasets or generate sample data.")
    st.stop()

# =====================================================
# FINAL CLEANING
# =====================================================

for col in merged_df.columns:
    if col.lower() == "date":
        continue
    
    if merged_df[col].dtype == "object":
        merged_df[col] = (
            merged_df[col]
            .astype(str)
            .str.replace("micromol/m2", "", regex=False)
            .str.replace("µmol/m2", "", regex=False)
            .str.strip()
        )

    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df = merged_df.ffill().bfill()

# =====================================================
# COLUMN DETECTION
# =====================================================

pollution_cols = identify_pollution_columns(merged_df)
wind_cols = identify_wind_columns(merged_df)
temperature_cols = identify_temperature_columns(merged_df)
numeric_cols = get_numeric_columns(merged_df)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Fusion & KPIs",
    "📈 Correlation Analytics",
    "📉 Lag & Significance",
    "🤖 Pattern & Prediction"
])

# =====================================================
# TAB 1: DATA FUSION & KPI DASHBOARD
# =====================================================

with tab1:
    st.header("📊 Data Fusion & KPI Dashboard")
    st.success(f"Successfully merged {len(merged_df):,} records.")

    # =================================================
    # DATA PREVIEW
    # =================================================

    st.subheader("Merged Dataset Preview")
    st.dataframe(merged_df.head(20), use_container_width=True)
    st.markdown("---")

    # =================================================
    # KPI SECTION
    # =================================================

    st.subheader("📌 Key Performance Indicators")

    try:
        kpis = calculate_basic_kpis(merged_df)
        available_metrics = list(kpis.keys())

        if len(available_metrics) >= 4:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    available_metrics[0],
                    round(kpis[available_metrics[0]]["mean"], 2)
                )

            with c2:
                st.metric(
                    available_metrics[1],
                    round(kpis[available_metrics[1]]["mean"], 2)
                )

            with c3:
                st.metric(
                    available_metrics[2],
                    round(kpis[available_metrics[2]]["mean"], 2)
                )

            with c4:
                st.metric(
                    available_metrics[3],
                    round(kpis[available_metrics[3]]["mean"], 2)
                )
        else:
            st.info(f"Available metrics: {', '.join(available_metrics)}")

    except Exception as e:
        st.error(f"KPI generation failed: {e}")

    st.markdown("---")

    # =================================================
    # DATASET INFO
    # =================================================

    st.subheader("📋 Dataset Information")

    i1, i2, i3 = st.columns(3)

    with i1:
        st.metric("Rows", merged_df.shape[0])

    with i2:
        st.metric("Columns", merged_df.shape[1])

    with i3:
        st.metric("Numeric Variables", len(numeric_cols))

    st.markdown("---")

    # =================================================
    # MISSING VALUES
    # =================================================

    st.subheader("🔍 Missing Values Audit")

    missing_df = pd.DataFrame({
        "Column": merged_df.columns,
        "Missing Values": merged_df.isna().sum(),
        "Missing %": (merged_df.isna().mean() * 100).round(2)
    })

    st.dataframe(missing_df, use_container_width=True)
    st.markdown("---")

    # =================================================
    # TEMPORAL COVERAGE
    # =================================================

    st.subheader("📅 Temporal Coverage")

    if "date" in merged_df.columns:
        d1, d2 = st.columns(2)

        with d1:
            st.metric("Start Date", str(merged_df["date"].min().date()))

        with d2:
            st.metric("End Date", str(merged_df["date"].max().date()))

    st.markdown("---")

    # =================================================
    # DESCRIPTIVE STATISTICS
    # =================================================

    st.subheader("📈 Descriptive Statistics")

    try:
        st.dataframe(
            merged_df.describe(include="all").transpose(),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Statistics failed: {e}")

    st.markdown("---")

    # =================================================
    # FEATURE DISTRIBUTION
    # =================================================

    st.subheader("📊 Feature Distribution Explorer")

    if len(numeric_cols) > 0:
        selected_feature = st.selectbox(
            "Select Variable",
            numeric_cols,
            key="distribution_feature"
        )

        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(merged_df[selected_feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_feature}")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Distribution plot failed: {e}")
    else:
        st.warning("No numeric columns available for distribution.")

    st.markdown("---")

    # =================================================
    # TIME SERIES EXPLORER
    # =================================================

    st.subheader("📈 Time Series Explorer")

    if len(numeric_cols) > 0 and "date" in merged_df.columns:
        ts_variable = st.selectbox(
            "Choose Variable",
            numeric_cols,
            key="time_series"
        )

        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(merged_df["date"], merged_df[ts_variable])
            ax.set_title(f"{ts_variable} Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel(ts_variable)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Time series plot failed: {e}")
    else:
        st.warning("Date column or numeric variables not available for time series.")

    st.markdown("---")

    # =================================================
    # STRONGEST CORRELATIONS
    # =================================================

    st.subheader("🔥 Strongest Correlations")

    try:
        strongest_corr_df = strongest_correlations(merged_df, top_n=20)
        
        if not strongest_corr_df.empty:
            st.dataframe(strongest_corr_df, use_container_width=True)
        else:
            st.info("No correlations found.")

    except Exception as e:
        st.error(f"Correlation extraction failed: {e}")

# =====================================================
# TAB 2: CORRELATION ANALYTICS
# =====================================================

with tab2:
    st.header("📈 Correlation Analytics")
    st.markdown(
        """
        Explore relationships between environmental variables.

        Features:
        • Correlation Heatmap
        • Pearson Correlation
        • Spearman Correlation
        • Scatter Plot Analysis
        • Trendline Detection
        • Automated Interpretation
        """
    )
    st.markdown("---")

    # =================================================
    # CORRELATION MATRIX
    # =================================================

    st.subheader("🔥 Correlation Heatmap")

    try:
        corr_matrix = calculate_correlation_matrix(merged_df)

        if not corr_matrix.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Environmental Correlation Matrix")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No numeric data available for correlation.")

    except Exception as e:
        st.error(f"Heatmap generation failed: {e}")

    st.markdown("---")

    # =================================================
    # VARIABLE SELECTOR
    # =================================================

    st.subheader("🔍 Variable Relationship Explorer")

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_variable = st.selectbox(
                "Select X Variable",
                numeric_cols,
                key="corr_x"
            )

        with col2:
            y_variable = st.selectbox(
                "Select Y Variable",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="corr_y"
            )

        st.markdown("---")

        # =================================================
        # PEARSON CORRELATION
        # =================================================

        pearson_result = None

        try:
            pearson_result = pearson_analysis(merged_df, x_variable, y_variable)

            if pearson_result:
                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Pearson r", round(pearson_result["correlation"], 4))

                with c2:
                    st.metric("Pearson p-value", round(pearson_result["p_value"], 6))
            else:
                st.warning("Unable to calculate Pearson correlation.")

        except Exception as e:
            st.error(f"Pearson analysis failed: {e}")

        st.markdown("---")

        # =================================================
        # SPEARMAN CORRELATION
        # =================================================

        try:
            spearman_result = spearman_analysis(merged_df, x_variable, y_variable)

            if spearman_result:
                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Spearman ρ", round(spearman_result["correlation"], 4))

                with c2:
                    st.metric("Spearman p-value", round(spearman_result["p_value"], 6))
            else:
                st.warning("Unable to calculate Spearman correlation.")

        except Exception as e:
            st.error(f"Spearman analysis failed: {e}")

        st.markdown("---")

        # =================================================
        # SCATTER PLOT
        # =================================================

        st.subheader("📊 Scatter Plot With Trendline")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(
                data=merged_df,
                x=x_variable,
                y=y_variable,
                scatter_kws={"alpha": 0.7},
                line_kws={"linewidth": 2},
                ax=ax
            )
            ax.set_title(f"{x_variable} vs {y_variable}")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Scatter plot failed: {e}")

        st.markdown("---")

        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader("🧠 Automated Interpretation")

        try:
            if pearson_result:
                corr = abs(pearson_result["correlation"])

                if corr >= 0.80:
                    strength = "Very Strong"
                elif corr >= 0.60:
                    strength = "Strong"
                elif corr >= 0.40:
                    strength = "Moderate"
                elif corr >= 0.20:
                    strength = "Weak"
                else:
                    strength = "Very Weak"

                direction = "Positive" if pearson_result["correlation"] > 0 else "Negative"
                significance = (
                    "Statistically Significant"
                    if pearson_result["p_value"] < 0.05
                    else "Not Statistically Significant"
                )

                st.info(
                    f"""
Relationship Strength: {strength}

Direction: {direction}

Statistical Result: {significance}
"""
                )
        except Exception as e:
            st.error(f"Interpretation failed: {e}")

        st.markdown("---")

    else:
        st.warning("At least two numeric columns are required.")

    # =================================================
    # TOP CORRELATIONS
    # =================================================

    st.subheader("🏆 Strongest Correlations")

    try:
        top_corr = strongest_correlations(merged_df, top_n=20)

        if not top_corr.empty:
            st.dataframe(top_corr, use_container_width=True)
        else:
            st.info("No strong correlations found.")

    except Exception as e:
        st.error(f"Strong correlation extraction failed: {e}")

    st.markdown("---")

    # =================================================
    # POLLUTION RELATIONSHIP ANALYSIS
    # =================================================

    st.subheader("🌫 Pollution Relationship Analysis")

    try:
        if len(pollution_cols) > 0 and len(numeric_cols) > 0:
            pollution_results = []

            for pollution in pollution_cols:
                for variable in numeric_cols:
                    if variable == pollution:
                        continue

                    result = pearson_analysis(merged_df, variable, pollution)

                    if result:
                        pollution_results.append({
                            "Pollution Variable": pollution,
                            "Related Variable": variable,
                            "Pearson_r": round(result["correlation"], 4),
                            "P_Value": round(result["p_value"], 6)
                        })

            if len(pollution_results) > 0:
                pollution_df = pd.DataFrame(pollution_results)
                pollution_df = pollution_df.sort_values("Pearson_r", key=abs, ascending=False)
                st.dataframe(pollution_df, use_container_width=True)
            else:
                st.info("No pollution relationships detected.")
        else:
            st.warning("Insufficient pollution or numeric columns for analysis.")

    except Exception as e:
        st.error(f"Pollution analysis failed: {e}")

# =====================================================
# TAB 3: LAG ANALYSIS & SIGNIFICANCE TESTING
# =====================================================

with tab3:
    st.header("📉 Lag Analysis & Statistical Significance")
    st.markdown(
        """
        Evaluate delayed environmental impacts.

        Features:
        • Weather vs Pollution Significance
        • Lag Correlation Analysis
        • Impact Matrix
        • Statistical Validation
        • Environmental Interpretation
        """
    )
    st.markdown("---")

    # =================================================
    # WEATHER VARIABLES
    # =================================================

    weather_variables = wind_cols + temperature_cols

    # =================================================
    # SIGNIFICANCE TESTING
    # =================================================

    st.subheader("🧪 Weather Impact Significance Testing")

    significance_df = pd.DataFrame()

    try:
        if len(weather_variables) > 0 and len(pollution_cols) > 0:
            significance_df = weather_pollution_significance(
                merged_df,
                weather_variables,
                pollution_cols
            )

            if not significance_df.empty:
                st.dataframe(significance_df, use_container_width=True)
            else:
                st.warning("No significance results available.")
        else:
            st.warning("Insufficient weather or pollution variables for significance testing.")

    except Exception as e:
        st.error(f"Significance testing failed: {e}")

    st.markdown("---")

    # =================================================
    # SUMMARY CARDS
    # =================================================

    st.subheader("📊 Significance Summary")

    try:
        if not significance_df.empty:
            significant_count = (
                significance_df[
                    significance_df["Significance"] == "Statistically Significant"
                ].shape[0]
            )

            total_tests = significance_df.shape[0]
            not_significant = total_tests - significant_count

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Tests", total_tests)

            with c2:
                st.metric("Significant", significant_count)

            with c3:
                st.metric("Not Significant", not_significant)

    except Exception as e:
        st.error(f"Summary card failed: {e}")

    st.markdown("---")

    # =================================================
    # LAG ANALYSIS
    # =================================================

    st.subheader("⏳ Lag Correlation Analysis")

    lag_results = pd.DataFrame()

    try:
        if len(weather_variables) > 0 and len(pollution_cols) > 0:
            lag_results = lag_correlation_analysis(
                merged_df,
                weather_variables,
                pollution_cols,
                lags=[1, 2]
            )

            if not lag_results.empty:
                st.dataframe(lag_results, use_container_width=True)
            else:
                st.warning("No lag relationships found.")
        else:
            st.warning("Insufficient variables for lag analysis.")

    except Exception as e:
        st.error(f"Lag analysis failed: {e}")

    st.markdown("---")

    # =================================================
    # LAG VISUALIZATION
    # =================================================

    st.subheader("📈 Lag Correlation Visualization")

    try:
        if not lag_results.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=lag_results, x="Lag", y="Correlation", hue="Target", ax=ax)
            ax.set_title("Lag Correlation Strength")
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Lag visualization failed: {e}")

    st.markdown("---")

    # =================================================
    # TOP LAG RELATIONSHIPS
    # =================================================

    st.subheader("🏆 Strongest Lag Relationships")

    try:
        if not lag_results.empty:
            strongest_lags = lag_results.copy()
            strongest_lags["Absolute_Correlation"] = strongest_lags["Correlation"].abs()
            strongest_lags = (
                strongest_lags.sort_values("Absolute_Correlation", ascending=False).head(15)
            )
            st.dataframe(strongest_lags, use_container_width=True)

    except Exception as e:
        st.error(f"Strong lag extraction failed: {e}")

    st.markdown("---")

    # =================================================
    # IMPACT MATRIX
    # =================================================

    st.subheader("🌍 Environmental Impact Matrix")

    try:
        if not significance_df.empty:
            impact_matrix = significance_df.pivot_table(
                index="Weather Variable",
                columns="Pollution Variable",
                values="Pearson_r"
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                impact_matrix,
                annot=True,
                cmap="RdYlGn",
                center=0,
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Weather vs Pollution Impact Matrix")
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Impact matrix failed: {e}")

    st.markdown("---")

    # =================================================
    # STRONGEST SIGNIFICANT RELATIONSHIP
    # =================================================

    st.subheader("🧠 Automated Statistical Interpretation")

    try:
        if not significance_df.empty:
            strongest_row = significance_df.iloc[
                significance_df["Pearson_r"].abs().idxmax()
            ]

            st.info(
                f"""
Strongest Relationship Found

Weather Variable: {strongest_row['Weather Variable']}

Pollution Variable: {strongest_row['Pollution Variable']}

Pearson Correlation: {strongest_row['Pearson_r']}

P-Value: {strongest_row['P_Value']}

Result: {strongest_row['Significance']}
"""
            )

    except Exception as e:
        st.error(f"Interpretation failed: {e}")

# =====================================================
# TAB 4: PATTERN RECOGNITION & PREDICTION
# =====================================================

with tab4:
    st.header("🤖 Pattern Recognition & Prediction Sandbox")
    st.markdown(
        """
        Advanced Analytics Modules

        • KMeans Clustering
        • Pattern Recognition
        • Anomaly Detection
        • Environmental Event Detection
        • Machine Learning Forecasting
        • Feature Importance
        • Future Predictions
        """
    )
    st.markdown("---")

    # =================================================
    # KMEANS PATTERN RECOGNITION
    # =================================================

    st.subheader("🎯 Pattern Recognition")

    try:
        # Prepare clean numeric data
        numeric_df = merged_df.drop(
            columns=['date'],
            errors='ignore'
        ).select_dtypes(include=np.number)

        numeric_df = numeric_df.dropna()

        if numeric_df.shape[1] > 0 and numeric_df.shape[0] >= 4:
            pattern_results = build_pattern_recognition_pipeline(
                numeric_df,
                n_clusters=4
            )

            clustered_df = pattern_results["clustered_data"]
            cluster_profiles = pattern_results["cluster_profiles"]

            st.success("Pattern recognition completed.")
            st.dataframe(cluster_profiles, use_container_width=True)

        else:
            st.warning(
                "Insufficient numeric features or data available for clustering. "
                "Need at least 4 rows and 1 numeric column."
            )

    except Exception as e:
        st.error(f"Pattern recognition failed: {e}")

    st.markdown("---")

    # =================================================
    # CLUSTER VISUALIZATION
    # =================================================

    st.subheader("📊 Cluster Visualization")

    try:
        if clustered_df is not None and not clustered_df.empty:
            numeric_cluster_cols = (
                clustered_df.select_dtypes(include=np.number).columns.tolist()
            )

            numeric_cluster_cols = [
                col for col in numeric_cluster_cols if col != "Cluster"
            ]

            if len(numeric_cluster_cols) >= 2:
                x_axis = st.selectbox(
                    "Cluster X Variable",
                    numeric_cluster_cols,
                    key="cluster_x"
                )

                y_axis = st.selectbox(
                    "Cluster Y Variable",
                    numeric_cluster_cols,
                    index=min(1, len(numeric_cluster_cols) - 1),
                    key="cluster_y"
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=clustered_df,
                    x=x_axis,
                    y=y_axis,
                    hue="Cluster",
                    palette="tab10",
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Need at least 2 numeric columns for visualization.")

    except Exception as e:
        st.error(f"Cluster chart failed: {e}")

    st.markdown("---")

    # =================================================
    # CLUSTER DISTRIBUTION
    # =================================================

    st.subheader("📈 Cluster Distribution")

    try:
        if clustered_df is not None and not clustered_df.empty:
            distribution = calculate_cluster_distribution(clustered_df)

            if not distribution.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(
                    distribution["Percentage"],
                    labels=distribution["Cluster"],
                    autopct="%1.1f%%"
                )
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Distribution failed: {e}")

    st.markdown("---")

    # =================================================
    # ANOMALY DETECTION
    # =================================================

    st.subheader("🚨 Anomaly Detection")

    try:
        if pattern_results is not None:
            anomaly_df = pattern_results["anomaly_data"]
            anomalies = get_top_anomalies(anomaly_df, n=15)

            if not anomalies.empty:
                st.write(f"Detected {len(anomalies)} anomalies.")
                st.dataframe(anomalies, use_container_width=True)
            else:
                st.info("No anomalies detected.")

    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")

    st.markdown("---")

    # =================================================
    # ENVIRONMENTAL EVENTS
    # =================================================

    st.subheader("🌍 Environmental Event Detection")

    try:
        numeric_events_df = merged_df.select_dtypes(include=np.number)

        if not numeric_events_df.empty:
            events = environmental_event_detector(numeric_events_df)

            if not events.empty:
                st.dataframe(events.head(20), use_container_width=True)
            else:
                st.info("No major environmental events detected.")
        else:
            st.warning("No numeric data available for event detection.")

    except Exception as e:
        st.error(f"Event detection failed: {e}")

    st.markdown("---")

    # =================================================
    # FORECASTING
    # =================================================

    st.subheader("📈 Predictive Forecasting")

    forecast_target = None
    forecast_results = None

    try:
        if len(pollution_cols) > 0:
            forecast_target = st.selectbox(
                "Forecast Target",
                pollution_cols
            )

            if st.button("Train Forecasting Model", key="forecast_button"):
                try:
                    forecast_results = train_forecasting_model(
                        merged_df,
                        forecast_target
                    )

                    model = forecast_results["model"]
                    metrics = forecast_results["metrics"]
                    prediction_df = forecast_results["prediction_df"]
                    feature_columns = forecast_results["feature_columns"]

                    st.success("Model trained successfully!")

                except Exception as e:
                    st.error(f"Model training failed: {e}")
        else:
            st.warning("No pollution columns available for forecasting.")

    except Exception as e:
        st.error(f"Forecasting setup failed: {e}")

    st.markdown("---")

    # =================================================
    # MODEL METRICS
    # =================================================

    try:
        if forecast_results and metrics:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("R²", round(metrics["R2"], 3))

            with c2:
                st.metric("MAE", round(metrics["MAE"], 3))

            with c3:
                st.metric("RMSE", round(metrics["RMSE"], 3))

    except Exception as e:
        st.error(f"Metric display failed: {e}")

    st.markdown("---")

    # =================================================
    # ACTUAL VS PREDICTED
    # =================================================

    st.subheader("📉 Actual vs Predicted")

    try:
        if forecast_results and prediction_df is not None and not prediction_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(prediction_df["Actual"].values, label="Actual")
            ax.plot(prediction_df["Predicted"].values, label="Predicted")
            ax.legend()
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Prediction plot failed: {e}")

    st.markdown("---")

    # =================================================
    # FEATURE IMPORTANCE
    # =================================================

    st.subheader("⭐ Feature Importance")

    try:
        if forecast_results and model is not None:
            importance_df = get_feature_importance(model, feature_columns)

            if not importance_df.empty:
                st.dataframe(importance_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=importance_df.head(10),
                    x="Importance",
                    y="Feature",
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Importance analysis failed: {e}")

    st.markdown("---")

    # =================================================
    # FUTURE FORECAST
    # =================================================

    st.subheader("🔮 Future Forecast")

    try:
        if forecast_results and forecast_target and len(pollution_cols) > 0:
            forecast_df = forecast_future_values(
                merged_df,
                forecast_target,
                periods=30
            )

            if not forecast_df.empty:
                st.dataframe(forecast_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(forecast_df["Date"], forecast_df["Forecast"])
                ax.set_title(f"30 Day Forecast: {forecast_target}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Future forecasting failed: {e}")

    st.markdown("---")

    # =================================================
    # DOWNLOAD DATA
    # =================================================

    st.subheader("⬇ Download Results")

    try:
        csv = merged_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Merged Dataset",
            data=csv,
            file_name="urban_environmental_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Download failed: {e}")

    st.markdown("---")

# =====================================================
# END OF APPLICATION
# =====================================================

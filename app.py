"""Streamlit dashboard for Americhem demand forecasting with Nixta TimeGEN-1.

Upload an Excel extract with the required columns to generate 12-month forecasts
by Plant and SBG. The dashboard shows historical demand, forecasts with
confidence intervals, accuracy metrics, and interpretability insights.
"""
import io
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    EXPECTED_COLUMNS,
    ForecastConfig,
    ForecastResult,
    filter_orders,
    load_orders,
    run_forecast_pipeline,
)

st.set_page_config(page_title="Americhem Demand Forecasting", layout="wide")
st.title("Americhem Demand Forecasting with Nixta TimeGEN-1")

st.markdown(
    """
    Upload an Excel file with the order history to generate a 12-month demand
    forecast by Plant and SBG using Nixta TimeGEN-1. The dashboard provides
    interactive filtering, tables, charts, confidence intervals, and accuracy
    plus interpretability metrics.
    """
)


def render_dataset_preview(df: pd.DataFrame) -> None:
    st.subheader("Data preview")
    st.dataframe(df.head(50))
    st.caption("Showing up to the first 50 rows of the uploaded dataset.")


@st.cache_data(show_spinner=False)
def _load_orders_cached(file_buffer: io.BytesIO) -> pd.DataFrame:
    return load_orders(file_buffer)


def forecast_to_table(results: List[ForecastResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        table = result.forecast.copy()
        table["Plant"] = result.plant
        table["SBG"] = result.sbg
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_forecast(history: pd.Series, result: ForecastResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Historical", mode="lines+markers"))
    fig.add_trace(
        go.Scatter(
            x=result.forecast["ds"],
            y=result.forecast["yhat"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([result.forecast["ds"], result.forecast["ds"][::-1]]),
            y=pd.concat([result.forecast["yhat_upper"], result.forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% CI",
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Month",
        yaxis_title="Order Quantity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_metrics(result: ForecastResult) -> None:
    cols = st.columns(3)
    cols[0].metric("MAE", f"{result.accuracy.get('MAE', float('nan')):.2f}" if result.accuracy else "N/A")
    cols[1].metric("MAPE", f"{result.accuracy.get('MAPE', float('nan')):.2%}" if result.accuracy else "N/A")
    cols[2].metric("RMSE", f"{result.accuracy.get('RMSE', float('nan')):.2f}" if result.accuracy else "N/A")

    st.caption("Accuracy is estimated from a hold-out of the most recent months before forecasting.")

    interp = result.interpretability
    if interp:
        st.markdown("**Interpretability metrics**")
        interp_cols = st.columns(2)
        interp_cols[0].metric("Trend strength", f"{interp.get('trend_strength', float('nan')):.2f}")
        interp_cols[1].metric("Seasonality strength", f"{interp.get('seasonality_strength', float('nan')):.2f}")
        st.caption("Trend/seasonality strength derived from seasonal decomposition of the historical series.")


def main() -> None:
    uploaded = st.file_uploader("Upload Excel orders", type=["xlsx", "xls"])
    if not uploaded:
        st.info("Waiting for an Excel file with the required fields to start forecasting.")
        st.write("Required columns:")
        st.code("\n".join(EXPECTED_COLUMNS))
        return

    try:
        df = _load_orders_cached(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load file: {exc}")
        return

    filters = st.expander("Filters", expanded=True)
    with filters:
        plants = st.multiselect("Plant", sorted(df["Plant"].unique()))
        sbgs = st.multiselect("SBG", sorted(df["SBG"].unique()))
        min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
        date_range = st.date_input("Order date range", value=(min_date, max_date))

    filtered_df = filter_orders(df, plants, sbgs, date_range if isinstance(date_range, tuple) else None)

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    render_dataset_preview(filtered_df)

    st.subheader("Forecast configuration")
    horizon = st.slider("Forecast horizon (months)", min_value=6, max_value=24, value=12)
    holdout = st.slider("Hold-out months for accuracy", min_value=1, max_value=6, value=3)

    if st.button("Generate forecast", type="primary"):
        with st.spinner("Running TimeGEN-1 forecasting pipeline..."):
            results = run_forecast_pipeline(filtered_df, ForecastConfig(horizon=horizon, holdout=holdout))

        if not results:
            st.warning("No forecast results generated. Check the data and filters.")
            return

        forecast_table = forecast_to_table(results)
        st.subheader("Forecast table")
        st.dataframe(forecast_table)
        csv_bytes = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast.csv")

        st.subheader("Plant/SBG level forecasts")
        for result in results:
            history = (
                filtered_df[(filtered_df["Plant"] == result.plant) & (filtered_df["SBG"] == result.sbg)]
                .set_index("Month")["Order Quantity"]
                .resample("MS")
                .sum()
            )
            st.markdown(f"### Plant {result.plant} — SBG {result.sbg}")
            st.plotly_chart(plot_forecast(history, result), use_container_width=True)
            render_metrics(result)

            with st.expander("Show input records", expanded=False):
                st.dataframe(
                    filtered_df[(filtered_df["Plant"] == result.plant) & (filtered_df["SBG"] == result.sbg)]
                )


if __name__ == "__main__":
    main()

# Americhem Demand Forecasting Dashboard

Interactive Streamlit dashboard that ingests Americhem order history from Excel
and generates 12-month demand forecasts by Plant and SBG using the Nixta
TimeGEN-1 model (with an in-app fallback when the model is not available).

## Features
- Excel upload with validation of all required order fields.
- Interactive filters for Plant, SBG, and order date range.
- Monthly aggregation of order quantities and 12-month forecasts by Plant/SBG.
- Plotly visualizations with confidence intervals and historical demand overlay.
- Accuracy (MAE, MAPE, RMSE) using a configurable hold-out window.
- Interpretability metrics (trend and seasonality strength from decomposition).
- Downloadable forecast table.

## Running locally
1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Upload an Excel file containing the following columns:
   ```
   Plant
   Order Date
   Order #
   Line #
   Product
   Description
   Global ID
   Cust No.
   Ship To
   Sales Rep
   SBU
   SBG
   Revised Ship Date
   Line Type
   Line Status
   Order Line UOM
   Order Line UOM2
   Order Quantity
   Amount
   On Hold?
   Order OEM
   Customer Product No.
   Month
   ```

4. Configure filters and forecasting horizon within the app to generate charts,
   tables, confidence intervals, and metrics.

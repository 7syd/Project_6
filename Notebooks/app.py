
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# Load saved model and store metadata
bundle = joblib.load("models/sales_app_bundle.pkl")
store_info = joblib.load("models/store_info.pkl")

model = bundle["model"]
features = bundle["features"]

st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="📊",
    layout="wide"
)

st.title("Rossmann Store Sales Forecast")
st.write("Predict daily sales using store information, dates, promotions and holidays.")

# Sidebar inputs
st.sidebar.header("Forecast Configuration")

store_ids = sorted(store_info["Store"].unique())

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_ids
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2015, 8, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date(2015, 8, 7)
)

promo = st.sidebar.selectbox("Promotion Active?", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday?", [0, 1])

state_holiday = st.sidebar.selectbox(
    "State Holiday",
    ["0", "a", "b", "c"],
    format_func=lambda x: {
        "0": "None",
        "a": "Public Holiday",
        "b": "Easter Holiday",
        "c": "Christmas"
    }[x]
)

# Validate date range
if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

if (end_date - start_date).days > 41:
    st.error("Please select a maximum forecast period of 42 days.")
    st.stop()

# Retrieve selected store metadata
store_row = store_info.loc[
    store_info["Store"] == selected_store
].iloc[0]

# Generate prediction input rows
dates = pd.date_range(start_date, end_date, freq="D")
prediction_rows = []

for current_date in dates:
    row = {
        "Store": selected_store,
        "DayOfWeek": current_date.dayofweek + 1,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "CompetitionDistance": store_row["CompetitionDistance"],
        "CompetitionOpenSinceMonth": store_row["CompetitionOpenSinceMonth"],
        "CompetitionOpenSinceYear": store_row["CompetitionOpenSinceYear"],
        "Promo2": store_row["Promo2"],
        "Promo2SinceWeek": store_row["Promo2SinceWeek"],
        "Promo2SinceYear": store_row["Promo2SinceYear"],
        "Year": current_date.year,
        "Month": current_date.month,
        "Day": current_date.day,
        "WeekOfYear": int(current_date.isocalendar().week),
        "IsWeekend": int(current_date.dayofweek >= 5),
        "IsMonthStart": int(current_date.is_month_start),
        "IsMonthEnd": int(current_date.is_month_end),
        "StoreType": store_row["StoreType"],
        "Assortment": store_row["Assortment"],
        "StateHoliday": state_holiday,
        "PromoInterval": store_row["PromoInterval"]
    }

    prediction_rows.append(row)

input_data = pd.DataFrame(prediction_rows)[features]

# Prediction
if st.sidebar.button("Generate Forecast", type="primary"):
    predictions = np.maximum(model.predict(input_data), 0)

    results = pd.DataFrame({
        "Date": dates,
        "Predicted_Sales": predictions.round(2)
    })

    st.subheader(f"Forecast for Store {selected_store}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predicted Sales",
        f"{results['Predicted_Sales'].sum():,.0f}"
    )

    col2.metric(
        "Average Daily Sales",
        f"{results['Predicted_Sales'].mean():,.0f}"
    )

    col3.metric(
        "Forecast Days",
        len(results)
    )

    st.subheader("Daily Sales Forecast")
    st.line_chart(
        results.set_index("Date")["Predicted_Sales"]
    )

    st.subheader("Prediction Details")
    st.dataframe(results, use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"store_{selected_store}_forecast.csv",
        mime="text/csv"
    )

st.caption(
    "Note: Promo and holiday selections are applied uniformly across the "
    "selected dates. Predictions are estimates from the trained Extra Trees model."
)

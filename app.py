"""
Late Delivery Risk Predictor
DataLab Analytics Final Project — Olist Brazilian E-commerce
Loads the trained XGBoost model + preprocessing bundle and predicts the
probability that a new order will be delivered late.

DEPLOYMENT LAYOUT (for Streamlit Community Cloud):
  app.py                          <- this file
  requirements.txt
  models/late_delivery_bundle.pkl <- copy of the bundle saved from your notebook
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import date

st.set_page_config(page_title="Late Delivery Predictor", page_icon="📦", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("models/late_delivery_bundle.pkl")


bundle = load_bundle()
model = bundle['model']
scaler = bundle['scaler']
encoder = bundle['encoder']
categorical_cols = bundle['categorical_cols']
full_feature_columns = bundle['full_feature_columns']
selected_features = bundle['selected_features']
zip_coords = bundle['zip_coords']


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_holiday_count(purchase_date, estimated_date):
    """Counts Brazilian public holidays falling within the delivery window."""
    years = list(range(purchase_date.year, estimated_date.year + 1))
    br_holidays = holidays.Brazil(years=years)
    holiday_dates = [pd.Timestamp(d) for d in br_holidays.keys()]
    start, end = pd.Timestamp(purchase_date), pd.Timestamp(estimated_date)
    return sum(start <= h <= end for h in holiday_dates)


st.title("📦 Late Delivery Risk Predictor")
st.caption("DataLab Analytics Final Project — Olist E-commerce")
st.write("Enter order details below to predict the probability that this order will be delivered late.")

main_categories = sorted(encoder.categories_[categorical_cols.index('main_category')])
customer_states = sorted(encoder.categories_[categorical_cols.index('customer_state')])
seller_states = sorted(encoder.categories_[categorical_cols.index('seller_state')])
payment_types = sorted(encoder.categories_[categorical_cols.index('payment_type')])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Order Details")
    purchase_date = st.date_input("Purchase date", value=date(2018, 6, 1))
    estimated_date = st.date_input("Estimated delivery date", value=date(2018, 6, 20))
    num_items = st.number_input("Number of items", min_value=1, value=1)
    main_category = st.selectbox("Product category", main_categories)
    payment_type = st.selectbox("Payment type", payment_types)
    payment_installments = st.number_input("Payment installments", min_value=1, value=1)
    payment_value = st.number_input("Payment value (R$)", min_value=0.0, value=100.0)

with col2:
    st.subheader("Location & Package")
    customer_state = st.selectbox("Customer state", customer_states)
    seller_state = st.selectbox("Seller state", seller_states)
    customer_zip = st.number_input("Customer zip code prefix", min_value=1000, max_value=99999, value=1310)
    seller_zip = st.number_input("Seller zip code prefix", min_value=1000, max_value=99999, value=14940)
    total_price = st.number_input("Total item price (R$)", min_value=0.0, value=150.0)
    total_freight_value = st.number_input("Total freight cost (R$)", min_value=0.0, value=20.0)
    total_weight_g = st.number_input("Total package weight (g)", min_value=1.0, value=1000.0)
    avg_length_cm = st.number_input("Package length (cm)", min_value=1.0, value=20.0)
    avg_height_cm = st.number_input("Package height (cm)", min_value=1.0, value=10.0)
    avg_width_cm = st.number_input("Package width (cm)", min_value=1.0, value=15.0)

if st.button("Predict Delivery Risk", type="primary"):
    if estimated_date <= purchase_date:
        st.error("Estimated delivery date must be after the purchase date.")
    else:
        cust_row = zip_coords[zip_coords['geolocation_zip_code_prefix'] == customer_zip]
        sell_row = zip_coords[zip_coords['geolocation_zip_code_prefix'] == seller_zip]

        if cust_row.empty or sell_row.empty:
            st.warning("Exact zip prefix not found in reference data — using an approximate distance instead.")
            distance_km = 430.0  # approx. median seller-customer distance observed in training data
        else:
            distance_km = haversine(
                cust_row['lat'].values[0], cust_row['lng'].values[0],
                sell_row['lat'].values[0], sell_row['lng'].values[0]
            )

        estimated_window_days = (estimated_date - purchase_date).days
        holidays_in_window = get_holiday_count(purchase_date, estimated_date)

        raw_input = {
            'num_items': num_items,
            'num_sellers': 1,
            'total_price': total_price,
            'avg_item_price': total_price / num_items,
            'total_freight_value': total_freight_value,
            'num_distinct_categories': 1,
            'total_weight_g': total_weight_g,
            'avg_weight_g': total_weight_g / num_items,
            'avg_length_cm': avg_length_cm,
            'avg_height_cm': avg_height_cm,
            'avg_width_cm': avg_width_cm,
            'payment_value': payment_value,
            'payment_installments': payment_installments,
            'holidays_in_delivery_window': holidays_in_window,
            'seller_customer_distance_km': distance_km,
            'purchase_month': purchase_date.month,
            'estimated_delivery_window_days': estimated_window_days
        }

        cat_input = pd.DataFrame([{
            'main_category': main_category,
            'customer_state': customer_state,
            'seller_state': seller_state,
            'payment_type': payment_type
        }])[categorical_cols]

        encoded = pd.DataFrame(
            encoder.transform(cat_input),
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        numeric_df = pd.DataFrame([raw_input])
        full_row = pd.concat([numeric_df, encoded], axis=1)
        full_row = full_row.reindex(columns=full_feature_columns, fill_value=0)

        scaled_row = pd.DataFrame(scaler.transform(full_row), columns=full_feature_columns)
        final_row = scaled_row[selected_features]

        probability = model.predict_proba(final_row)[0][1]

        st.divider()
        st.subheader("Prediction Result")
        st.metric("Probability of Late Delivery", f"{probability * 100:.2f}%")

        if probability >= 0.5:
            st.error(f"⚠️ The model predicts a {probability * 100:.2f}% probability that this order will be delivered late.")
        else:
            st.success(f"✅ The model predicts a {probability * 100:.2f}% probability that this order will be delivered late (likely on-time).")

        st.caption("Model: XGBoost | Trained on Olist Brazilian E-commerce data | DataLab Analytics Final Project")

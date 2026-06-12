import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Cab Cancellation Predictor", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

st.title("🚖 Cab Cancellation Prediction")
st.markdown("""
Predict whether a cab booking will be canceled or not based on booking details.
This tool helps in operational efficiency and resource allocation.
""")

# Sidebar for inputs
st.sidebar.header("User Input Features")

def user_input_features():
    vehicle_model_id = st.sidebar.number_input("Vehicle Model ID", min_value=0, value=12)
    from_area_id = st.sidebar.number_input("From Area ID", min_value=0, value=393)
    from_lat = st.sidebar.number_input("From Latitude", value=12.91)
    from_long = st.sidebar.number_input("From Longitude", value=77.61)
    to_lat = st.sidebar.number_input("To Latitude", value=13.02)
    to_long = st.sidebar.number_input("To Longitude", value=77.70)
    booking_lead_time = st.sidebar.slider("Booking Lead Time (Hours)", 0.0, 500.0, 24.0)
    
    travel_type = st.sidebar.selectbox("Travel Type ID", [1, 2, 3], index=1)
    online_booking = st.sidebar.selectbox("Online Booking", [0, 1], index=0)
    mobile_site_booking = st.sidebar.selectbox("Mobile Site Booking", [0, 1], index=0)

    data = {
        'vehicle_model_id': vehicle_model_id,
        'from_area_id': from_area_id,
        'from_lat': from_lat,
        'from_long': from_long,
        'to_lat': to_lat,
        'to_long': to_long,
        'booking_lead_time': booking_lead_time,
        'travel_type_id_2': 1 if travel_type == 2 else 0,
        'travel_type_id_3': 1 if travel_type == 3 else 0,
        'online_booking_1': 1 if online_booking == 1 else 0,
        'mobile_site_booking_1': 1 if mobile_site_booking == 1 else 0
    }
    
    features = pd.DataFrame(data, index=[0])
    # Reorder columns to match model_columns
    features = features[model_columns]
    return features

df_input = user_input_features()

# Display input summary
st.subheader("Booking Details Summary")
st.write(df_input)

# Prediction
if st.button("Predict Cancellation"):
    # Scale input
    scaled_input = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ **The booking is likely to be CANCELED.**")
    else:
        st.success("✅ **The booking is likely to be COMPLETED.**")
    
    st.subheader("Prediction Probability")
    st.write(f"Confidence (Canceled): {prediction_proba[1]:.2%}")
    st.write(f"Confidence (Completed): {prediction_proba[0]:.2%}")

st.markdown("---")
st.write("Developed for Cab Cancellation Project Deployment.")

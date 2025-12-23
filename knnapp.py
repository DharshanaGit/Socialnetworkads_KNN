import streamlit as st
import numpy as np
import joblib as jb

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = jb.load("Social_network_ads.pkl")
scaler = jb.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KNN Purchase Prediction", layout="centered")

st.title("🛒 Social Network Ads Prediction")
st.write("Predict whether a user will purchase a product.")

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=10000, max_value=200000, value=50000)

# Predict button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # Output
    if prediction == 1:
        st.success("✅ Prediction: Purchased")
    else:
        st.error("❌ Prediction: Not Purchased")

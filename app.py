import streamlit as st
import joblib
import numpy as np

st.title("Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn.")

# Load the model
model = joblib.load('churn_model.pkl')

# Inputs
total_charges = st.number_input("Total Charges (₹)", min_value=0.0, value=1000.0)
monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=100.0)
tenure = st.slider("Tenure (months)", 0, 72, 12)

if st.button("Predict"):
    x = np.array([[total_charges, monthly_charges, tenure]])
    result = model.predict(x)[0]
    st.success("Customer likely to stay." if result == 0 else "Customer likely to churn.")








import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Churn Prediction App")
st.write("Enter customer details to predict churn risk.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
streaming_tv = st.selectbox("Streaming TV?", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies?", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)

# Convert inputs to encoded numeric format
input_dict = {
    'gender': 1 if gender == "Male" else 0,
    'StreamingTV': 1 if streaming_tv == "Yes" else 0,
    'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
    'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
    'MonthlyCharges': monthly_charges
}

input_df = pd.DataFrame([input_dict])

# Scale numeric features
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to churn — Risk: {prob:.2%}")
    else:
        st.success(f"✅ Likely to stay — Risk: {prob:.2%}")

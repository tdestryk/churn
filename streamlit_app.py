# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# 💾 Load model and scaler
model = joblib.load('model/churn_lgb_model.pkl')
scaler = joblib.load('model/lgb_scaler.pkl')

# 🖼️ Page Setup
st.set_page_config(
    page_title="Streaming Churn Predictor",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎥 Streaming App Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to cancel their subscription.")

# 📋 Sidebar Inputs
st.sidebar.header("📋 Customer Info")
user_input = {
    'gender': st.sidebar.selectbox("Gender", ['Male', 'Female']),
    'SeniorCitizen': st.sidebar.selectbox("Senior Citizen", ['No', 'Yes']),
    'Contract': st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year']),
    'InternetService': st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No']),
    'StreamingTV': st.sidebar.selectbox("Streaming TV", ['Yes', 'No']),
    'OnlineSecurity': st.sidebar.selectbox("Online Security", ['Yes', 'No']),
    'MonthlyCharges': st.sidebar.slider("Monthly Charges", 0, 150, 70),
    'TotalCharges': st.sidebar.slider("Total Charges", 0, 9000, 2000)
}

# Default values for missing training features
default_features = {
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingMovies': 'No',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check'
}
user_input.update(default_features)

# 🔤 Encoding
encode_map = {
    'gender': {'Female': 0, 'Male': 1},
    'SeniorCitizen': {'No': 0, 'Yes': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'Yes': 1},
    'OnlineBackup': {'No': 0, 'Yes': 1},
    'DeviceProtection': {'No': 0, 'Yes': 1},
    'TechSupport': {'No': 0, 'Yes': 1},
    'StreamingTV': {'No': 0, 'Yes': 1},
    'StreamingMovies': {'No': 0, 'Yes': 1},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
}

for col, mapping in encode_map.items():
    user_input[col] = mapping.get(user_input[col], 0)

# 🧮 Create DataFrame
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]
df_input = pd.DataFrame([user_input])[feature_order]

# 🔍 Predict
if st.button("🔎 Predict Churn"):
    df_input_scaled = scaler.transform(df_input)
    proba = model.predict_proba(df_input_scaled)[0][1]

    st.subheader("🔢 Prediction Results")
    st.metric("Churn Probability", f"{proba:.2%}")

    if proba >= 0.4:
        st.error("⚠️ This customer is at **high risk** of churning.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.markdown("---")
    st.caption("Made with ❤️ using LightGBM and Streamlit.")


# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb

# 💾 Load model and scaler
model = joblib.load('model/churn_lgb_model.pkl')
scaler = joblib.load('model/lgb_scaler.pkl')

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🎬 Streaming App Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to cancel their subscription.")

# 📋 Sidebar Inputs
st.sidebar.header("📋 Customer Info")
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
streaming_tv = st.sidebar.selectbox("Streaming TV", ['Yes', 'No'])
online_security = st.sidebar.selectbox("Online Security", ['Yes', 'No'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ['No', 'Yes'])

# 🎛️ Encode Inputs
encode_map = {
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'StreamingTV': {'No': 0, 'Yes': 1},
    'OnlineSecurity': {'No': 0, 'Yes': 1},
    'gender': {'Female': 0, 'Male': 1}
}

# Selected + Default Input Data
user_input = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Contract': contract,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'StreamingTV': streaming_tv,
    'MonthlyCharges': monthly_charges,
    
    # Fill remaining features with default values
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingMovies': 'No',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'TotalCharges': monthly_charges * 12  # Reasonable default estimate
}

# Encode categorical values
for col, mapping in encode_map.items():
    user_input[col] = mapping[user_input[col]]

# Fill in the remaining encoded features
encode_map.update({
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1},
    'OnlineBackup': {'No': 0, 'Yes': 1},
    'DeviceProtection': {'No': 0, 'Yes': 1},
    'TechSupport': {'No': 0, 'Yes': 1},
    'StreamingMovies': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
})

# Apply encoding
for col, mapping in encode_map.items():
    user_input[col] = mapping.get(user_input[col], 0)

# Convert to DataFrame
df_input = pd.DataFrame([user_input])

# Ensure exact column match
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

df_input = df_input[feature_order]  # Reorder and enforce all columns
df_input_scaled = scaler.transform(df_input)


# 🔍 Prediction
if st.button("🧠 Predict Churn"):
    proba = model.predict_proba(df_input_scaled)[0][1]
    st.subheader("🔎 Results")
    st.metric("Churn Probability", f"{proba:.2%}")

    if proba >= 0.4:
        st.error("⚠️ This customer is at **high risk** of churning.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.markdown("---")
    st.caption("Made with ❤️ using LightGBM and Streamlit.")

# Customer Churn Prediction with Streamlit

This project uses a machine learning model to predict customer churn based on a few key features such as gender, contract type, streaming services, and monthly charges. It includes:

- A trained Random Forest model
- A simplified interface built with Streamlit
- Visualizations for model performance and feature importance

---

## 🔍 What is Churn?

**Customer churn** occurs when a customer stops doing business with a company. Predicting churn allows companies to take action before losing customers. 
Customer churn in the context of streaming apps refers to when a user cancels their subscription or stops using the platform. Predicting churn helps streaming services identify at-risk users early—so they can take proactive steps like offering personalized recommendations, discounts, or improved support to retain those users before they leave.

---

## 📦 Features Used

The model was trained using these 5 core features:
- `gender`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `MonthlyCharges`

These were selected for a lightweight, user-friendly prediction interface.

---

## 🚀 How to Run the Streamlit App

### 1. Clone the repository
```bash
git clone https://github.com/tdestryk/churn-predictor.git
cd churn-predictor

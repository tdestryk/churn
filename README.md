# 📉 Customer Churn Prediction

This project builds a machine learning model to predict customer churn using a real-world telecom dataset. It demonstrates end-to-end data science skills, including EDA, feature engineering, model training, evaluation, and saving the model for deployment.

---

## 📁 Project Structure
## 📊 Dataset Overview

- Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Size: ~7,000 records
- Target: `Churn` (Yes/No)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights:
- 📆 Customers on **month-to-month contracts** churn far more than those on long-term contracts
- 💰 Higher **MonthlyCharges** are linked to higher churn
- 🧓 Seniors and those without **OnlineSecurity** or **TechSupport** churn more frequently

---

## 🤖 Modeling

- Model: `RandomForestClassifier`
- Features: Cleaned + label encoded
- Scoring: Classification report + confusion matrix

---

## 📦 Output

- ✅ Trained model saved as `model/churn_model.pkl` using `joblib`

---

## 🧪 Try It Yourself

To run:

```bash
pip install -r requirements.txt
jupyter notebook
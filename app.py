# app.py
import streamlit as st
import numpy as np
import pickle
import os

# Load models
@st.cache_resource
def load_models():
    with open("clf_model.sav", "rb") as f:
        clf_model = pickle.load(f)
    with open("reg_model.sav", "rb") as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

@st.cache_data
def load_feature_means():
    if os.path.exists("feature_means.pkl"):
        with open("feature_means.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return {}

clf_model, reg_model = load_models()
feature_means = load_feature_means()

# Full features used by the original StockMarketPredictor
full_feature_names = [
    'Volume', 'Returns', 'Log_Returns', 'Volatility', 'MA_5', 'MA_20', 'RSI',
    'GDP', 'UNEMPLOYMENT', 'INFLATION', 'FED_RATE', 'VIX', '10Y_TREASURY', '3M_TREASURY',
    'CONSUMER_SENTIMENT', 'INDUSTRIAL_PRODUCTION', 'HOUSING_STARTS', 'RETAIL_SALES',
    'M2_MONEY_SUPPLY', 'INFLATION_RATE', 'YIELD_CURVE',
    'GDP_lag1', 'GDP_lag7', 'UNEMPLOYMENT_lag1', 'UNEMPLOYMENT_lag7',
    'INFLATION_lag1', 'INFLATION_lag7', 'FED_RATE_lag1', 'FED_RATE_lag7',
    'VIX_lag1', 'VIX_lag7', '10Y_TREASURY_lag1', '10Y_TREASURY_lag7',
    '3M_TREASURY_lag1', '3M_TREASURY_lag7', 'CONSUMER_SENTIMENT_lag1',
    'CONSUMER_SENTIMENT_lag7', 'INDUSTRIAL_PRODUCTION_lag1', 'INDUSTRIAL_PRODUCTION_lag7',
    'HOUSING_STARTS_lag1', 'HOUSING_STARTS_lag7', 'RETAIL_SALES_lag1', 'RETAIL_SALES_lag7',
    'M2_MONEY_SUPPLY_lag1', 'M2_MONEY_SUPPLY_lag7', 'INFLATION_RATE_lag1', 'INFLATION_RATE_lag7',
    'YIELD_CURVE_lag1', 'YIELD_CURVE_lag7'
]

# Top 10 most important features for UI input
ui_feature_names = [
    'Log_Returns', 'Returns', 'VIX_lag7', 'RSI', 'MA_20', 'Volume',
    'Volatility', 'VIX_lag1', 'MA_5', 'VIX'
]

# App UI
st.title("📈 Stock Market Predictor")
st.sidebar.title("🔍 Choose Prediction Type")
prediction_type = st.sidebar.radio("Model", ("Classification", "Regression"))

st.sidebar.markdown("---")
st.sidebar.info("Enter top feature values below")

# Collect UI inputs
input_dict = {}
for name in ui_feature_names:
    default_val = feature_means.get(name, 0.0)
    val = st.number_input(name, value=float(default_val), format="%.4f")
    input_dict[name] = val

# Fill missing features with mean values
final_feature_vector = [input_dict.get(name, feature_means.get(name, 0.0)) for name in full_feature_names]
features = np.array([final_feature_vector])

# Prediction
if st.button("Predict"):
    if prediction_type == "Classification":
        prediction = clf_model.predict(features)[0]
        st.success(f"📊 Predicted Market Direction: {'UP' if prediction == 1 else 'DOWN'}")

    elif prediction_type == "Regression":
        prediction = reg_model.predict(features)[0]
        st.success(f"📈 Predicted Next Day Return: {prediction:.4f} ({prediction*100:.2f}%)")

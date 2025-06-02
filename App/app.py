import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and selected features
model = joblib.load('cancer_model.pkl')

scaler = joblib.load('scaler.pkl')
top_features = joblib.load('selected_features.pkl')

# Set Streamlit app title
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter values for the following features to predict whether the tumor is malignant or benign:")

# Input form
user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"{feature}", format="%.4f")

# When the button is clicked
if st.button("Predict"):
    # Convert inputs into DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.success("🎉 The model predicts **Benign** (Non-cancerous).")
    else:
        st.error("⚠️ The model predicts **Malignant** (Cancerous).")
    
    st.write("Prediction Probability:")
    st.write(f"Malignant: {prediction_proba[0]:.2f}, Benign: {prediction_proba[1]:.2f}")

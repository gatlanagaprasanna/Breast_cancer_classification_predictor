import os
import streamlit as st
import numpy as np
import joblib

# Load the model
model_path = os.path.join(os.path.dirname(__file__),'breast_cancer_model.pkl')
model=joblib.load(model_path)

st.title("Breast Cancer Prediction App")
st.markdown("Enter the values for all 30 features below to predict whether the tumor is benign or malignant.")

# Feature input fields
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

input_values = []

# Collect all 30 inputs
for feature in features:
    value = st.number_input(f"Enter value for {feature}", format="%.5f")
    input_values.append(value)

# Predict on button click
if st.button("Predict"):
    input_array = np.array([input_values])  # shape must be (1, 30)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ The tumor is **Malignant**.")
    else:
        st.success("✅ The tumor is **Benign**.")

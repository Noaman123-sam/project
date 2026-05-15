import streamlit as st
import pandas as pd
import joblib


model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
features_list = joblib.load('features_list.pkl')

st.title("❤️ Heart Disease Prediction")

# 
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 100, 50)
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
with col2:
    ap_hi = st.number_input("Systolic BP (ap_hi)", 50, 250, 120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", 30, 150, 80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])

if st.button("Predict"):
    # -----
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    high_blood_pressure = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
    age_chol_risk = age * cholesterol
    is_healthy = 1 if (cholesterol == 1 and gluc == 1 and high_blood_pressure == 0) else 0

    # 
    input_data = {
        'age': age,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'bmi': bmi,
        'pulse_pressure': pulse_pressure,
        'high_blood_pressure': high_blood_pressure,
        'age_chol_risk': age_chol_risk,
        'is_healthy': is_healthy,
        'chol_1': 1 if cholesterol == 1 else 0,
        'chol_2': 1 if cholesterol == 2 else 0,
        'chol_3': 1 if cholesterol == 3 else 0,
        'gluc_1': 1 if gluc == 1 else 0
    }

 
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features_list] # 

    #
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ High Risk Detected")
    else:
        st.success("✅ Low Risk Detected")

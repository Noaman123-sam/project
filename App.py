import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")
st.write("Please fill in the patient's medical details below:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=1, max_value=100, value=50)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    gluc = st.selectbox("Glucose Level", [1, 2, 3], help="1: Normal, 2: Above Normal, 3: High")

with col2:
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], help="1: Normal, 2: Above Normal, 3: High")

st.markdown("---")


if st.button("Predict Health Status", use_container_width=True):
    with st.spinner('Analyzing data...'):
    
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        high_blood_pressure = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
        age_chol_risk = age * cholesterol
        is_healthy = 1 if (cholesterol == 1 and gluc == 1 and high_blood_pressure == 0) else 0

        input_dict = {
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

        input_df = pd.DataFrame([input_dict])

       
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0]

            st.subheader("Result:")
            if prediction[0] == 1:
                st.error(f"⚠️ High Risk Detected")
                st.write(f"The model is **{probability[1]*100:.2f}%** confident that there is a heart condition.")
            else:
                st.success(f"✅ Low Risk Detected")
                st.write(f"The model is **{probability[0]*100:.2f}%** confident that the heart is healthy.")
                
        except Exception as e:
            st.error(f"Error in prediction: {e}")

st.info("Note: This is an AI-based prediction and should not replace professional medical advice.")
import streamlit as st
import pandas as pd
import joblib

# 1.
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
features_list = joblib.load('features_list.pkl')

# 2.  (Dark UI Customization)
st.set_page_config(page_title="Cardio AI - Dark Edition", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    /* تغيير خلفية التطبيق بالكامل */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* تنسيق الكروت (الخلفية اللي ورا المدخلات) */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
    }

    /* تنسيق زرار التحليل */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(45deg, #ff4b4b, #ff7575);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }

    /* العناوين */
    h1, h2, h3 {
        color: #ff4b4b !important;
    }

    /* تظبيط شكل الـ Metric */
    [data-testid="stMetricValue"] {
        color: #ff4b4b;
        font-size: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 
st.title("❤️ Cardio Guard AI")
st.markdown("#### *Advanced Deep Analytics for Heart Health*")
st.write("---")

# 
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Bio Data")
    age = st.number_input("Age", 1, 100, 50)
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)

with col2:
    st.markdown("### 🩺 Vitals")
    ap_hi = st.number_input("Systolic BP", 50, 250, 120)
    ap_lo = st.number_input("Diastolic BP", 30, 150, 80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
    gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])

st.write("")
st.write("")

# 4. 
if st.button("START SYSTEM DIAGNOSTIC"):
    # 13
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    high_blood_pressure = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
    age_chol_risk = age * cholesterol
    is_healthy = 1 if (cholesterol == 1 and gluc == 1 and high_blood_pressure == 0) else 0

    input_dict = {
        'age': age, 'weight': weight, 'ap_hi': ap_hi, 'ap_lo': ap_lo, 'bmi': bmi,
        'pulse_pressure': pulse_pressure, 'high_blood_pressure': high_blood_pressure,
        'age_chol_risk': age_chol_risk, 'is_healthy': is_healthy,
        'chol_1': 1 if cholesterol == 1 else 0, 'chol_2': 1 if cholesterol == 2 else 0,
        'chol_3': 1 if cholesterol == 3 else 0, 'gluc_1': 1 if gluc == 1 else 0
    }

    df = pd.DataFrame([input_dict])[features_list]
    
    with st.spinner('Accessing AI Neural Engine...'):
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)[0]

        st.write("---")
        if prediction[0] == 1:
            st.error("## 🚨 High Risk Detected")
            st.metric("Risk Probability", f"{prob[1]*100:.1f}%")
            st.info("Clinical intervention is recommended.")
        else:
            st.success("## ✅ Healthy Heart")
            st.metric("Confidence Score", f"{prob[0]*100:.1f}%")
            st.balloons() #

st.markdown("<br><br><p style='text-align: center; color: gray;'>V 2.5 | Secure Medical AI Protocol</p>", unsafe_allow_html=True)

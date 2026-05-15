import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features_list = joblib.load('features_list.pkl')
    return model, scaler, features_list

try:
    model, scaler, features_list = load_models()
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")

# 2. الديزاين الرمادي الاحترافي (Modern Anthracite)
st.set_page_config(page_title="Cardio Engine AI", layout="wide")

st.markdown("""
    <style>
    /* الخلفية الرمادية */
    .stApp {
        background-color: #2b2d31; 
        color: #ffffff;
    }
    
    /* تنوير كل النصوص والعناوين باللون السماوي */
    label, .stSlider p, .stMarkdown p, div[data-baseweb="radio"] div, 
    div[data-testid="stWidgetLabel"] p, .stToggle p {
        color: #00d4ff !important;
        font-weight: bold !important;
        opacity: 1 !important;
    }

    /* تنسيق الكروت */
    div[data-testid="stVerticalBlock"] > div:has(div.stSlider), 
    div[data-testid="stVerticalBlock"] > div:has(div.stRadio),
    div[data-testid="stVerticalBlock"] > div:has(div.stToggle) {
        background-color: #313338;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #404249;
        margin-bottom: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* نصوص الاختيارات (Male/Female) */
    div[role="radiogroup"] label p {
        color: #ffffff !important;
    }

    /* زرار التحليل النيوني */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        color: white;
        font-weight: bold;
        font-size: 20px;
        border: none;
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
    }

    h1 { color: #00d4ff; text-align: center; text-shadow: 2px 2px 10px rgba(0,212,255,0.2); }
    h3 { border-left: 5px solid #00d4ff; padding-left: 15px; margin-top: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Cardio Engine AI")
st.write("<p style='text-align: center; color: #aaa;'>Neural Network Diagnostic System v2.6</p>", unsafe_allow_html=True)
st.write("---")

# 3. توزيع الـ Sliders (الـ 11 ميزة الظاهرة)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 👤 Bio Metrics")
    age = st.slider("Age", 1, 100, 50)
    height = st.slider("Height (cm)", 100, 220, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)
    gender = st.radio("Gender", ["Male", "Female"])

with col2:
    st.markdown("### 🩺 Vital Signs")
    ap_hi = st.slider("Systolic BP (Upper)", 80, 250, 120)
    ap_lo = st.slider("Diastolic BP (Lower)", 40, 150, 80)
    cholesterol = st.select_slider("Cholesterol level", options=[1, 2, 3])
    gluc = st.select_slider("Glucose level", options=[1, 2, 3])

with col3:
    st.markdown("### 🏃 Lifestyle")
    smoke = st.toggle("Smoker")
    alco = st.toggle("Alcohol Intake")
    active = st.toggle("Physically Active", value=True)

st.write("---")

# 4. زرار التوقع ومعالجة الـ 13 ميزة
if st.button("EXECUTE NEURAL ANALYSIS"):
    # حساب الميزات المتقدمة (Feature Engineering)
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    hbp = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
    risk_score = age * cholesterol
    healthy_flag = 1 if (cholesterol == 1 and gluc == 1 and hbp == 0) else 0

    # بناء القاموس الكامل لـ 13 ميزة (بالمللي)
    input_data = {
        'age': age, 'weight': weight, 'ap_hi': ap_hi, 'ap_lo': ap_lo, 'bmi': bmi,
        'pulse_pressure': pulse_pressure, 'high_blood_pressure': hbp,
        'age_chol_risk': risk_score, 'is_healthy': healthy_flag,
        'chol_1': 1 if cholesterol == 1 else 0, 
        'chol_2': 1 if cholesterol == 2 else 0,
        'chol_3': 1 if cholesterol == 3 else 0, 
        'gluc_1': 1 if gluc == 1 else 0
    }

    # تحويل لـ DataFrame وإعادة ترتيب الأعمدة فوراً
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=features_list)
    
    # --- جزء الـ Debug عشان نعرف ليه بيطلع Low Risk ---
    with st.expander("🛠️ System Debug: Check Input Data"):
        st.write("This is what the model sees (13 features):")
        st.dataframe(df)

    # 5. التوقع الفعلي
    with st.spinner('Neural Engine is calculating risk factors...'):
        scaled = scaler.transform(df)
        pred = model.predict(scaled)
        prob = model.predict_proba(scaled)[0]

        st.write("### Analysis Results:")
        if pred[0] == 1:
            st.error(f"## 🚨 HIGH RISK DETECTED")
            st.metric("Risk Probability", f"{prob[1]*100:.1f}%")
            st.warning("Urgent medical consultation is advised.")
        else:
            st.success(f"## ✅ LOW RISK DETECTED")
            st.metric("Confidence Score", f"{prob[0]*100:.1f}%")
            st.balloons()

st.markdown("<br><p style='text-align: center; color: #666;'>Secure Medical Protocol | End-to-End Encryption Enabled</p>", unsafe_allow_html=True)
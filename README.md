# ❤️ Heart Disease Risk Prediction App

An interactive machine learning web app that predicts a patient's risk of cardiovascular disease based on key health indicators, built with **Scikit-learn** and deployed using **Streamlit**.

🔗 **Live Demo:** https://project-h67yd7tpkl8ecm6bfjtmhy.streamlit.app/
📂 **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) 

---

## 📌 Overview

This project builds an end-to-end machine learning pipeline to classify whether a patient is at risk of cardiovascular disease, based on clinical and demographic data such as age, blood pressure, cholesterol, and lifestyle habits. Several models were trained and compared, and the best-performing one was deployed through an interactive Streamlit interface for real-time predictions.

---

## 🗂️ Dataset

- **File:** `cardio_train.csv`
- **Records used:** ~67,500 patient records (54,027 train / 13,507 test)
- **Features used:** 13
- **Target variable:** `cardio` — presence (1) or absence (0) of cardiovascular disease
- **Key raw features:** Age, Gender, Height, Weight, Systolic & Diastolic blood pressure (`ap_hi`, `ap_lo`), Cholesterol, Glucose, Smoking, Alcohol intake, Physical activity

### Engineered Features
On top of the raw features, additional engineered features were created to boost model performance:
- `bmi` — Body Mass Index derived from height and weight
- `pulse_pressure` — difference between systolic and diastolic pressure
- `high_blood_pressure` — flag derived from blood pressure thresholds
- `age_chol_risk` — combined age and cholesterol risk indicator
- `chol_2`, `chol_3` — encoded cholesterol level categories

---

## ⚙️ Approach

1. **Data Preprocessing**
   - Cleaned and validated the raw `cardio_train.csv` data
   - Handled outliers in blood pressure and BMI-related fields
   - Scaled numerical features using `scaler.pkl`

2. **Feature Engineering**
   - Created derived features (BMI, pulse pressure, hypertension flag, age-cholesterol risk score)
   - Selected the final 13 features used for training (saved in `features_list.pkl`)

3. **Model Training & Comparison**
   Multiple classification models were trained and evaluated on the same train/test split:

   | Model | Accuracy |
   |---|---|
   | Logistic Regression | 72.58% |
   | Random Forest | 72.75% |
   | **XGBoost** | **72.85%** |
   | LightGBM (final model) | 72.75% |

4. **Final Model Evaluation** *(LightGBM)*

   | Metric | Class 0 (Healthy) | Class 1 (Sick) |
   |---|---|---|
   | Precision | 0.71 | 0.75 |
   | Recall | 0.78 | 0.67 |
   | F1-score | 0.74 | 0.71 |

   - **Overall Accuracy:** 72.75%
   - **Weighted F1-Score:** 71.11%
   - **Confusion Matrix:**

     |  | Predicted Healthy | Predicted Sick |
     |---|---|---|
     | **Actual Healthy** | 5296 | 1546 |
     | **Actual Sick** | 2135 | 4530 |

5. **Feature Importance**
   The strongest predictors of cardiovascular disease risk were:
   1. `ap_hi` (systolic blood pressure) — strongest predictor
   2. `high_blood_pressure` flag
   3. `ap_lo` (diastolic blood pressure)
   4. `pulse_pressure`
   5. `age_chol_risk`

6. **Deployment**
   - Built an interactive **Streamlit** app (`App.py`) that loads the trained model, scaler, and feature list to serve real-time predictions

---


## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost, LightGBM
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Environment:** Google Colab (Jupyter Notebook)

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Noaman123-sam/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run App.py
```

---

## 📁 Project Structure
```
heart-disease-prediction-ml/
│
├── App.py                    # Streamlit application
├── final_project.ipynb       # EDA, feature engineering & model training notebook
├── cardio_train.csv          # Dataset
├── heart_disease_model.pkl   # Trained ML model (LightGBM)
├── scaler.pkl                # Fitted feature scaler
├── features_list.pkl         # Selected feature list used by the model
├── requirements.txt          # Project dependencies
└── README.md
```

---

## 📈 Results & Insights
- Blood pressure-related features (`ap_hi`, `ap_lo`, `pulse_pressure`, and the engineered `high_blood_pressure` flag) were by far the strongest predictors of cardiovascular disease risk.
- All four models (Logistic Regression, Random Forest, XGBoost, LightGBM) performed within ~0.3% of each other in accuracy (~72.6–72.9%), suggesting the ceiling is largely set by the dataset's inherent noise rather than model choice.
- The final model recalls 78% of healthy patients correctly but only 67% of at-risk patients — meaning it's more conservative and prone to under-flagging risk, which is an important trade-off to communicate for a health-related use case.

---

## 🔮 Future Improvements
- Address the recall gap on the "Sick" class (e.g. class weighting, threshold tuning) since under-detecting risk is more costly in a health context
- Add SHAP explanations for individual predictions
- Improve UI/UX of the Streamlit app
- Deploy on a permanent hosting platform (Streamlit Cloud / Hugging Face Spaces)




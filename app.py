import streamlit as st 
import pandas as pd    
import joblib   

model = joblib.load("logistic_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Attack Prediction")
st.markdown("provide the following information to predict the risk of heart attack:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality(STA)", "Left Ventricular Hypertrophy(LVH)"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 10.0, 1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("The model predicts a high risk of heart attack.")
    else:
        st.success("The model predicts a low risk of heart attack.")

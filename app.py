import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("models/best_model.pkl")

# Path for inference logs
log_file = "data/inference_log.csv"

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict whether they are diabetic or not.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)

if st.button("Predict"):
    input_df = pd.DataFrame([[age, gender, bp, bmi]], 
                             columns=["age", "gender", "blood_pressure", "BMI"])
    prediction = model.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    st.success(f"🔮 Prediction: {result}")

    # 🔹 Save inference request
    input_df["prediction"] = prediction
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(log_file):
        input_df.to_csv(log_file, index=False)
    else:
        input_df.to_csv(log_file, mode="a", header=False, index=False)

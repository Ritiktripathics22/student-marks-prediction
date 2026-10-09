import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("student_marks_model.pkl")

# Website heading
st.title("Student Marks Prediction")
st.write("Predict student marks using study hours and attendance.")

# Input fields
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

# Prediction button
if st.button("Predict Marks"):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance]
    })

    predicted_marks = model.predict(new_student)[0]

    st.success(
        f"Predicted Marks: {predicted_marks:.2f}"
    )

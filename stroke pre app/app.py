import streamlit as st
import pandas as pd
import joblib

# Title and Styling
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #4CAF50; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Stroke Prediction App")
st.subheader("Predict the likelihood of stroke based on patient details.")
st.markdown("Provide the following details in the sidebar to get a prediction.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("lda_stroke_model.pkl")  # Ensure the file is in the same directory

model = load_model()

# Input Form - Sidebar
st.sidebar.title("Patient Details")
st.sidebar.info("Fill in the patient's details below:")

# Two-column layout for better organization
with st.sidebar:
    age = st.number_input("🧓 Age (in years)", min_value=1, max_value=120, step=1)
    gender = st.radio("⚥ Gender", ["Male", "Female"])
    hypertension = st.radio("💊 Hypertension", ["No", "Yes"])
    heart_disease = st.radio("❤️ Heart Disease", ["No", "Yes"])
    avg_glucose_level = st.slider(
        "🩸 Average Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, step=0.1
    )
    bmi = st.slider("⚖️ BMI (Body Mass Index)", min_value=0.0, max_value=50.0, step=0.1)

    # Additional Features
    smoking_status = st.selectbox(
        "🚬 Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )
    Residence_type = st.selectbox(
        "🏡 Residence Type", ["Urban", "Rural"]
    )
    work_type = st.selectbox(
        "💼 Work Type",
        ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"],
    )
    ever_married = st.radio("💍 Ever Married", ["No", "Yes"])

# Mapping input to model-compatible format
data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,  # Adjust based on your dataset encoding
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,  # Leave as-is if handled by encoder
    "Residence_type": Residence_type,  # Leave as-is if handled by encoder
    "work_type": work_type,  # Leave as-is if handled by encoder
    "ever_married": ever_married,  # Leave as-is if handled by encoder
}

# Convert to DataFrame
input_df = pd.DataFrame([data])

# Main Section
st.header("Prediction Results")

if st.button("🔍 Predict"):
    try:
        # Perform prediction
        prediction = model.predict(input_df)
        result = "🟢 No Stroke" if prediction[0] == 0 else "🔴 Stroke"
        st.success(f"The predicted result is: **{result}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

st.markdown("---")
st.markdown(
    """
    ### How Does It Work?
    This app uses a machine learning model to predict the likelihood of a stroke based on:
    - **Age**
    - **Health history** (hypertension, heart disease, BMI)
    - **Lifestyle factors** (smoking, work type, residence type)
    
    Please consult a medical professional for personalized advice.
    """
)


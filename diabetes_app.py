import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

# Title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics.
Adjust the sliders and click 'Predict' to see the result.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Input section on main page (no sidebar)
st.header("👤 Patient Details")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 2)
    blood_pressure = st.slider("Blood Pressure (mmHg)", 40, 120, 70)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    age = st.slider("Age", 21, 100, 30)

with col2:
    glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 120)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
    insulin = st.slider("Insulin Level (IU/mL)", 0, 846, 80)
    dpf = st.slider("Diabetes Pedigree Function", 0.08, 2.5, 0.5)

if st.button("Predict Diabetes Risk"):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High risk of diabetes")
    else:
        st.success("✅ Low risk of diabetes")

    st.write(f"**Probability of diabetes:** `{prediction_proba[1]*100:.2f}%`")

    # Show feature importance
    st.subheader("📊 Key Factors Influencing Prediction")
    coefficients = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(coefficients.set_index('Feature'))

# Dataset toggle options
if st.checkbox("Show raw data"):
    st.subheader("📂 Diabetes Dataset")
    st.write(df)

if st.checkbox("Show statistics"):
    st.subheader("📈 Data Statistics")
    st.write(df.describe())

# Instructions
st.markdown("""
---
### ℹ️ How to use:
1. Adjust the sliders with patient info  
2. Click **Predict Diabetes Risk**  
3. See the prediction and probabilities  
""")

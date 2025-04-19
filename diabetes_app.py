import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time  # For simulating progress
import plotly.graph_objects as go  # Ensure this import is correct

# Set page config - must be the first Streamlit command
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

# Inject custom CSS to hide the link icon next to headers
st.markdown("""
    <style>
        .css-1v3fvcr a {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Diabetes Prediction Web App")
st.markdown("""
This web app predicts the likelihood of diabetes based on health metrics.<br>
Adjust the options and click 'Predict' to see the result.
""", unsafe_allow_html=True)

# Function to estimate skin thickness based on BMI
def estimate_skin_thickness(bmi):
    skin_thickness = 0.2 * bmi  # Example: skin thickness = 0.2 * BMI
    return min(skin_thickness, 99)  # Cap to 99 (as per dataset range)

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

# Input widgets
st.header("👤 Patient Details")

pregnancies = st.slider("Pregnancies", 0, 17, 2)
glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 120)
blood_pressure = st.slider("Diastolic blood pressure (mmHg)", 40, 120, 70)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
insulin = st.slider("Insulin Level (IU/mL)", 0, 846, 80)
age = st.slider("Age", 21, 100, 30)

# Default skin thickness based on BMI
skin_thickness = estimate_skin_thickness(bmi)

# Warning about skin thickness measurement
st.warning("""
Using BMI to estimate skin thickness is a simplified approach and may slightly reduce the accuracy of diabetes risk predictions compared to using actual skinfold measurements. This trade-off is made to improve user-friendliness. For more precise results, actual clinical measurements should be used.
""")

# Add checkbox to show skin thickness warning and set manual thickness
show_skin_thickness_warning = st.checkbox("I know my skin thickness")

# If the checkbox is checked, let the user set skin thickness manually
if show_skin_thickness_warning:
    skin_thickness = st.slider("Skin Thickness (mm)", 0.0, 99.0, skin_thickness)

# Replace slider with radio buttons for Diabetes Pedigree Function
st.markdown("### Family History of Diabetes - A Questionnaire to determine Diabetes Pedigree Function (DPF) ")
family_history = st.radio(
    "Select your family history level:",
    [
        "No family history",
        "Some relatives (uncle, aunt, grandparents)",
        "Parents or siblings"
    ]
)

# Mapping user-friendly options to DPF values
dpf_mapping = {
    "No family history": 0.2,
    "Some relatives (uncle, aunt, grandparents)": 0.6,
    "Parents or siblings": 1.2
}
dpf = dpf_mapping[family_history]

# Explanation of DPF
st.markdown("""
<div style="background-color: #1E3A8A; padding: 10px; border-radius: 5px;">
    <strong>What is Diabetes Pedigree Function (DPF)?</strong><br>
    This score estimates genetic risk based on family history.<br>
    A higher score means closer relatives (like parents or siblings) have diabetes.<br>
    (Since this value can’t be self-measured easily, users can select an option above and our program automatically fixes a mean DPF value based on the user's input.)
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction button
if st.button("Predict Diabetes Risk"):
    # Initialize progress bar
    progress_bar = st.progress(0)
    
    # Simulate the process of model computation
    for i in range(1, 101):
        time.sleep(0.001)  # Sleep to simulate processing
        progress_bar.progress(i)
    
    # Run the prediction after the progress bar reaches 100%
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Display the results
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("⚠️ High risk of diabetes")
    else:
        st.success("✅ Low risk of diabetes")
    
    st.write(f"Probability of diabetes: {prediction_proba[1]*100:.2f}%")


  # Show model confidence using gauge chart
    st.subheader("Model Confidence")
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[1] * 100,  # Multiplying by 100 to make it percentage
            title={'text': "Diabetes Risk Confidence (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "red"},
                   'steps': [
                       {'range': [0, 50], 'color': "green"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "red"}],
                   'threshold': {
                       'line': {'color': "black", 'width': 4},
                       'thickness': 0.75,
                       'value': prediction_proba[1] * 100}}))
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating the chart: {e}")


    # Show feature importance
    st.subheader("Key Factors Influencing Prediction")
    coefficients = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(coefficients.set_index('Feature'))

# Show dataset info
if st.checkbox("Show raw data"):
    st.subheader("Diabetes Dataset")
    st.write(df)
    
if st.checkbox("Show statistics"):
    st.subheader("Data Statistics")
    st.write(df.describe())

st.markdown("""
### Reference
This project uses the Pima Indians Diabetes Dataset from Plotly.  
[Plotly Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
""")

st.markdown("""
<div style="text-align: center; font-size: 18px; color: #FF6347;">
    Made with ❤️ by Anas Ashik Ismail
</div>
""", unsafe_allow_html=True)

# 🩺 Diabetes Prediction Web App

A simple, interactive web application built using **Streamlit** that predicts the likelihood of diabetes based on user-input health metrics using **Logistic Regression**.

## 🔍 About

This app is designed to help users quickly assess their risk of diabetes by entering basic health data like glucose levels, BMI, age, etc. The prediction is based on the **Pima Indians Diabetes dataset** from the UCI Machine Learning Repository.

## 📦 Features

- 🎯 Predicts diabetes risk using machine learning (Logistic Regression)
- 📊 Probability of prediction shown with detailed output
- 🧠 Shows most important features influencing prediction
- 📁 Option to view raw data and statistics
- 📱 Mobile responsive UI with clean layout
- 👤 Patient info input with easy sliders

## 🚀 Try it out

You can try the app live at: [🌐 Diabetes Prediction by Anas](https://anasdiabetesprediction.streamlit.app)



## 🧪 Inputs

| Feature | Description | Range |
|--------|-------------|--------|
| Pregnancies | Number of times pregnant | 0 - 17 |
| Glucose | Plasma glucose concentration | 50 - 200 mg/dL |
| Blood Pressure | Diastolic blood pressure | 40 - 120 mmHg |
| Skin Thickness | Triceps skinfold thickness | 0 - 99 mm |
| Insulin | Serum insulin | 0 - 846 IU/mL |
| BMI | Body Mass Index | 10.0 - 60.0 |
| Diabetes Pedigree Function | Likelihood of diabetes based on family history | 0.08 - 2.5 |
| Age | Age of the patient | 21 - 100 |

## 🧠 What is Diabetes Pedigree Function?

This value is a **quantitative measure of genetic influence**—calculated using **family history** and **lineage information**. A higher value suggests stronger hereditary influence on diabetes risk. (Since this value can’t be self-measured easily, users can select an approximate value or be guided using options.)

## 🛠️ Setup Instructions

1. **Clone the repo**

<pre>
git clone https://github.com/your-username/diabetes-predictor-app.git
cd diabetes-predictor-app
</pre>
  

2. **Install required packages & Run**

<pre>
pip install -r requirements.txt
streamlit run diabetes_app.py
</pre>

## 📁 Dataset Source

[Pima Indians Diabetes Database (UCI/Plotly)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
 - A Standard Dataset used in Diabetes Prediction Models



## 🙌 Credits

 - Created with ❤️ by Anas Ashik Ismail
  
  - Built using Streamlit, Scikit-learn, and Pandas

## 📃 License

MIT License – use it, build on it, share it!









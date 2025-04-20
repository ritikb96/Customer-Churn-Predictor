import streamlit as st
import joblib
import pandas as pd
from src.cleanStrategy import DataPreprocessStrategy
import logging
import os
model_path = os.path.join("models", "model.pkl")
model = joblib.load(model_path)

st.title("Churn Prediction App")
st.write("Fill the form below to predict Churn!")

with st.form("my form"):
    customerID = st.text_input("customerID")
    gender = st.radio("gender",["Male","Female"])
    age = st.selectbox("seniorcitizen",["Yes","No"])
    partner = st.radio('Partner:',['Yes','No'])
    dependents = st.radio('Dependents',['Yes','No'])
    tenure = st.text_input('tenure')
    phoneservice = st.radio('Phoneservice',["Yes","No"])
    multipleLines = st.selectbox("MultipleLines",["No phone service","No","Yes"])
    InternetService = st.selectbox("InternetService",["DSL","Fiber Optics","No"])
    OnlineSecurity = st.selectbox("OnlineSecurity",["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup",["Yes","No","No internet service"])
    DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])
    TechSupport = st.selectbox("TechSupport",["Yes","No","No internet service"])
    StreamingTV = st.selectbox("StreamingTV",["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies",["Yes","No","No internet service"])
    Contract = st.selectbox("Contract",["Month-to-month","One year","Two Year"])
    PaperlessBilling = st.radio("PaperlessBilling",["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatice)","Credit card (automatic)"])
    MonthlyCharge = st.slider("MonthlyCharge",0,150)
    TotalCharge = st.text_input("Totalcharge")








    submit = st.form_submit_button("go")


if submit:
    input_data = pd.DataFrame([{
        'customerID': customerID,
        'gender':gender,
        'SeniorCitizen': 1 if age == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phoneservice,
        'MultipleLines': multipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup':OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharge,
        'TotalCharges': float(TotalCharge)
    }])
    
    try:
        input_data.head()
        preprocessor = DataPreprocessStrategy()

        #lets clean the input data
        cleaned_df = preprocessor.inference_cleaning(input_data)

        #now lets predict on the data
        prediction = model.predict(cleaned_df)

        #and lets display the data
        # Display results
        results = pd.DataFrame({
            'Prediction': ['Churn' if p == 1 else 'No Churn' for p in prediction]
        })
        st.write("Predictions:")
        st.dataframe(results)
    except Exception as e:
        st.error(f"Error processing the file:{e}")
        

    

    
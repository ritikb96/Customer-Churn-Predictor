# Customer Churn Prediction 🚀

Welcome to the **Customer Churn Prediction** project! This solution uses classification techniques to predict customer churn in the telecom industry. By analyzing customer data, we aim to identify factors that contribute to churn and help companies take proactive measures to retain customers.

## Overview 📊

Churn prediction is a vital part of customer relationship management. This project leverages machine learning techniques to classify whether a customer will churn based on their usage, contract type, and other behavioral factors. The dataset used is the **telecom dataset**, which contains key features related to customer activity and demographics.

This repository contains the following components:
- **Machine Learning Model**: Built to predict customer churn using classification algorithms.
- **MLflow Experiment Tracker**: A tool to track model performance, metrics, and experiments.
- **Streamlit Web App**: A user-friendly interface to interact with the model and upload data for predictions.

You can directly test the app here: https://customer-churn-predictor-dg9nzen8xeey7a7gfplcff.streamlit.app/

## Installation 🛠️

Before you can start working with the project, you need to install the required dependencies. To set up the environment, follow the steps below:

 **Clone the repository**:

```bash
git clone https://github.com/ritikb96/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

 **📁 Viewing Experiment Results with MLflow**

This project uses **MLflow** to track different churn prediction experiments — including model parameters, metrics, and training runs.

Once you've trained your models, you can explore the results with MLflow’s UI. Just run:

```bash
mlflow ui
```
🚀 How to Launch the App

1. Navigate to the `streamlit` directory:

```bash
cd streamlit-app
```
2. Run the app
```bash
streamlit run Upload.py
```

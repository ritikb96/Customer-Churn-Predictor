import streamlit as st
import pandas as pd
import joblib
import os
import sys
# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from steps.clean_data import clean_df
from src.cleanStrategy import DataPreprocessStrategy

st.set_page_config(page_title="UploadCSV", page_icon="📤")

model_path = os.path.join("models", "model.pkl")
model = joblib.load(model_path)

st.title("Churn Prediction App")
st.write("Upload a CSV file and predict customer churn.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read file
    data = pd.read_csv(uploaded_file)
    st.write("Raw data:", data.head())

    try:
        preprocessor = DataPreprocessStrategy()
        cleaned_data = preprocessor.inference_cleaning(data)



        # Only use X_test (or X_train if you're just uploading one batch)
        preds = model.predict(cleaned_data)

        # Display results
        results = pd.DataFrame({
            'Prediction': ['Churn' if p == 1 else 'No Churn' for p in preds]
        })
        st.write("Predictions:")
        st.dataframe(results)

    except Exception as e:
        st.error(f"Error processing file: {e}")
